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
      - ELLIPSOID: semi-axes (x, y, z)
      - CYLINDER: radius in x, half-height in y (axis +Z)
      - CONE: radius in x, half-height in y (axis +Z, apex at +Z)
    """

    shape_type: int
    scale: wp.vec3


@wp.func
def center_func(geom: GenericShapeData, data_provider: SupportMapDataProvider) -> wp.vec3:
    """
    Return the center of a primitive in its local frame.

    For most primitives, this is the origin (0,0,0), but for some shapes
    like cones, the geometric center differs from the coordinate origin.
    """
    if geom.shape_type == int(GeoType.CONE):
        # Cone centroid is at 1/4 of the total height above the base plane.
        # With base at z=-half_height and apex at z=+half_height, this is z = -half_height/2.
        half_height = geom.scale[1]
        return wp.vec3(0.0, 0.0, -0.5 * half_height)
    else:
        # For box, sphere, capsule, ellipsoid, cylinder: center is at origin
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
    - ELLIPSOID: semi-axes in x/y/z
    - CYLINDER: radius in x, half-height in y (axis along +Z)
    - CONE: radius in x, half-height in y (axis along +Z, apex at +Z)
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

        # Capsule = segment + sphere (adapted from C# code to Z-axis convention)
        # Sphere part: support in normalized direction
        if dir_len_sq > eps:
            n = wp.normalize(dir_safe)
        else:
            n = wp.vec3(1.0, 0.0, 0.0)
        result = n * radius

        # Segment endpoints are at (0, 0, +half_height) and (0, 0, -half_height)
        # Use sign of Z-component to pick the correct endpoint
        if dir_safe[2] >= 0.0:
            result = result + wp.vec3(0.0, 0.0, half_height)
            feature_id = 1  # top cap
        else:
            result = result + wp.vec3(0.0, 0.0, -half_height)
            feature_id = 2  # bottom cap

    elif geom.shape_type == int(GeoType.ELLIPSOID):
        # Ellipsoid support for semi-axes a, b, c in direction d:
        # p* = (a^2 dx, b^2 dy, c^2 dz) / sqrt((a dx)^2 + (b dy)^2 + (c dz)^2)
        a = geom.scale[0]
        b = geom.scale[1]
        c = geom.scale[2]
        if dir_len_sq > eps:
            adx = a * dir_safe[0]
            bdy = b * dir_safe[1]
            cdz = c * dir_safe[2]
            denom_sq = adx * adx + bdy * bdy + cdz * cdz
            if denom_sq > eps:
                denom = wp.sqrt(denom_sq)
                result = wp.vec3((a * a) * dir_safe[0] / denom, (b * b) * dir_safe[1] / denom, (c * c) * dir_safe[2] / denom)
            else:
                result = wp.vec3(a, 0.0, 0.0)
        else:
            result = wp.vec3(a, 0.0, 0.0)
        feature_id = 0

    elif geom.shape_type == int(GeoType.CYLINDER):
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Cylinder support: project direction to XY plane for lateral surface
        dir_xy = wp.vec3(dir_safe[0], dir_safe[1], 0.0)
        dir_xy_len_sq = wp.length_sq(dir_xy)

        if dir_xy_len_sq > eps:
            n_xy = wp.normalize(dir_xy)
            lateral_point = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, 0.0)
        else:
            lateral_point = wp.vec3(radius, 0.0, 0.0)

        # Choose between top cap, bottom cap, or lateral surface
        if dir_safe[2] > 0.0:
            result = wp.vec3(lateral_point[0], lateral_point[1], half_height)
            feature_id = 1  # top cap
        elif dir_safe[2] < 0.0:
            result = wp.vec3(lateral_point[0], lateral_point[1], -half_height)
            feature_id = 2  # bottom cap
        else:
            result = lateral_point
            feature_id = 0  # lateral surface

    elif geom.shape_type == int(GeoType.CONE):
        radius = geom.scale[0]
        half_height = geom.scale[1]

        # Cone support: apex at +Z, base disk at z=-half_height.
        # Using slope k = radius / (2*half_height), the optimal support is:
        #   apex if dz >= k * ||d_xy||, otherwise base rim in d_xy direction.
        apex = wp.vec3(0.0, 0.0, half_height)
        dir_xy = wp.vec3(dir_safe[0], dir_safe[1], 0.0)
        dir_xy_len = wp.length(dir_xy)
        k = radius / (2.0 * half_height) if half_height > eps else 0.0

        if dir_xy_len <= eps:
            # Purely vertical direction
            if dir_safe[2] >= 0.0:
                result = apex
                feature_id = 1  # apex
            else:
                result = wp.vec3(radius, 0.0, -half_height)
                feature_id = 2  # base edge
        else:
            if dir_safe[2] >= k * dir_xy_len:
                result = apex
                feature_id = 1  # apex
            else:
                n_xy = dir_xy / dir_xy_len
                result = wp.vec3(n_xy[0] * radius, n_xy[1] * radius, -half_height)
                feature_id = 2  # base edge

    else:
        # Unhandled type: return origin and feature 0
        result = wp.vec3(0.0, 0.0, 0.0)
        feature_id = 0

    return result, feature_id
