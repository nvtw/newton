# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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
import warp as wp

from .types import GeoType


@wp.struct
class SupportMapDataProvider:
    """
    Placeholder for data access needed by support mapping (e.g., mesh buffers).
    Extend with fields as required by your shapes.
    """

    pass


@wp.struct
class GenericShapeData:
    """
    Minimal shape descriptor for support mapping.

    Fields:
    - shape_type: matches values from GeoType
    - scale: parameter encoding per primitive
      - BOX: half-extents (x, y, z)
      - SPHERE: radius in x
      - CAPSULE: radius in x, half-height in y (axis +Z)
    """

    shape_type: int
    scale: wp.vec3


@wp.func
def center_func(geom: GenericShapeData, data_provider: SupportMapDataProvider) -> wp.vec3:
    """
    Return the center of a primitive in its local frame.
    """
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def support_map(
    geom: GenericShapeData, direction: wp.vec3, data_provider: SupportMapDataProvider
) -> tuple[wp.vec3, int]:
    """
    Return the support point of a primitive in its local frame, with a feature id.

    Conventions for `geom.scale`:
    - BOX: half-extents in x/y/z
    - SPHERE: radius in x component
    - CAPSULE: radius in x, half-height in y (axis along +Z)
    """

    # handle zero direction robustly
    eps = 1.0e-12
    dir_len_sq = wp.length_sq(direction)
    dir_safe = wp.vec3(1.0, 0.0, 0.0)
    if dir_len_sq > eps:
        dir_safe = direction

    result = wp.vec3(0.0, 0.0, 0.0)
    feature_id = int(0)

    if geom.shape_type == int(GeoType.BOX):
        sx = 1.0 if dir_safe[0] >= 0.0 else -1.0
        sy = 1.0 if dir_safe[1] >= 0.0 else -1.0
        sz = 1.0 if dir_safe[2] >= 0.0 else -1.0

        result = wp.vec3(sx * geom.scale[0], sy * geom.scale[1], sz * geom.scale[2])

        # Bit mask consistent with reference: x->4, y->2, z->1
        feature_id = 0
        if sx >= 0.0:
            feature_id |= 4
        if sy >= 0.0:
            feature_id |= 2
        if sz >= 0.0:
            feature_id |= 1

    elif geom.shape_type == int(GeoType.SPHERE):
        radius = geom.scale[0]
        if dir_len_sq > eps:
            n = wp.normalize(dir_safe)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius
        feature_id = 0

    elif geom.shape_type == int(GeoType.CAPSULE):
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # axis is +Z; segment support combined with spherical offset
        if dir_safe[2] > 0.0:
            base = wp.vec3(0.0, 0.0, half_height)
            feature_id = 1  # top cap
        elif dir_safe[2] < 0.0:
            base = wp.vec3(0.0, 0.0, -half_height)
            feature_id = 2  # bottom cap
        else:
            base = wp.vec3(0.0, 0.0, 0.0)
            feature_id = 0  # lateral band

        if dir_len_sq > eps:
            n = wp.normalize(dir_safe)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = base + n * radius

    else:
        # Unhandled type: return origin and feature 0
        result = wp.vec3(0.0, 0.0, 0.0)
        feature_id = 0

    return result, feature_id
