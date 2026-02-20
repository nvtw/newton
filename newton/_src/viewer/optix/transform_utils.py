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

"""Shared transform helpers for OptiX viewer modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def quaternion_to_matrix3(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion ``[x, y, z, w]`` to a 3x3 rotation matrix."""
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def build_transform_matrix(
    position: Iterable[float],
    rotation_xyzw: Iterable[float],
    scale: float | Iterable[float] = 1.0,
) -> np.ndarray:
    """Build a row-major 4x4 transform from position, quaternion, and scale."""
    px, py, pz = [float(v) for v in position]
    qx, qy, qz, qw = [float(v) for v in rotation_xyzw]
    if isinstance(scale, Iterable) and not isinstance(scale, (str, bytes)):
        sx, sy, sz = [float(v) for v in scale]
    else:
        s = float(scale)
        sx = sy = sz = s
    m = np.eye(4, dtype=np.float32)
    r = quaternion_to_matrix3(qx, qy, qz, qw)
    r[:, 0] *= sx
    r[:, 1] *= sy
    r[:, 2] *= sz
    m[:3, :3] = r
    m[:3, 3] = np.array([px, py, pz], dtype=np.float32)
    return m


def mat4_to_optix_transform12(m: np.ndarray) -> np.ndarray:
    """Convert a 4x4 matrix to OptiX 3x4 row-major transform array."""
    m = np.asarray(m, dtype=np.float32).reshape(4, 4)
    return np.array(
        [
            m[0, 0],
            m[0, 1],
            m[0, 2],
            m[0, 3],
            m[1, 0],
            m[1, 1],
            m[1, 2],
            m[1, 3],
            m[2, 0],
            m[2, 1],
            m[2, 2],
            m[2, 3],
        ],
        dtype=np.float32,
    )
