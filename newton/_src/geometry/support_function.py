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
from enum import IntEnum

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


class IntersectionResult(IntEnum):
    NO_INTERSECTION = 0  # No intersection found
    INTERSECTION = 1  # Intersection found
    NOT_SUPPORTED = 2


@wp.func
def ray_sphere_intersection(ro: wp.vec3, rd: wp.vec3, radius: float) -> tuple[int, wp.vec3]:
    oc = ro
    b = wp.dot(oc, rd)
    qc = oc - b * rd
    h = radius * radius - wp.dot(qc, qc)
    if h < 0.0:
        return int(IntersectionResult.NO_INTERSECTION), wp.vec3(0.0, 0.0, 0.0)
    h = wp.sqrt(h)
    t1 = -b - h
    t2 = -b + h
    t = t1 if wp.abs(t1) < wp.abs(t2) else t2
    return int(IntersectionResult.INTERSECTION), ro + t * rd

@wp.func
def ray_capsule_intersection(ro: wp.vec3, rd: wp.vec3, radius: float, half_height: float) -> tuple[int, wp.vec3]:
    pa = wp.vec3(0.0, 0.0, -half_height)
    pb = wp.vec3(0.0, 0.0, half_height)
    ba = pb - pa
    oa = ro - pa

    t = wp.dot(oa, ba) / wp.dot(ba, ba)
    t = wp.clamp(t, 0.0, 1.0)

    closest_on_axis = pa + t * ba
    to_ro = ro - closest_on_axis
    dist = wp.length(to_ro)

    if dist < 1.0e-8:
        projected = closest_on_axis + wp.vec3(radius, 0.0, 0.0)
    else:
        projected = closest_on_axis + (to_ro / dist) * radius

    return int(IntersectionResult.INTERSECTION), projected




@wp.func
def ray_cylinder_intersection(ro: wp.vec3, rd: wp.vec3, radius: float, half_height: float) -> tuple[int, wp.vec3]:
    a = wp.vec3(0.0, 0.0, -half_height)
    b = wp.vec3(0.0, 0.0, half_height)
    ba = b - a
    oc = ro - a
    baba = wp.dot(ba, ba)
    bard = wp.dot(ba, rd)
    baoc = wp.dot(ba, oc)
    k2 = baba - bard * bard
    k1 = baba * wp.dot(oc, rd) - baoc * bard
    k0 = baba * wp.dot(oc, oc) - baoc * baoc - radius * radius * baba
    h = k1 * k1 - k2 * k0
    t_best = 1.0e20
    found = False
    if h >= 0.0:
        h_sqrt = wp.sqrt(h)
        for i in range(2):
            t = (-k1 - h_sqrt) / k2 if i == 0 else (-k1 + h_sqrt) / k2
            y = baoc + t * bard
            if y > 0.0 and y < baba:
                if wp.abs(t) < wp.abs(t_best):
                    t_best = t
                    found = True
        for cap_y in range(2):
            t_cap = ((0.0 if cap_y == 0 else baba) - baoc) / bard
            if wp.abs(k1 + k2 * t_cap) < h_sqrt:
                if wp.abs(t_cap) < wp.abs(t_best):
                    t_best = t_cap
                    found = True
    if not found:
        return int(IntersectionResult.NO_INTERSECTION), wp.vec3(0.0, 0.0, 0.0)
    return int(IntersectionResult.INTERSECTION), ro + t_best * rd


@wp.func
def ray_cone_intersection(ro: wp.vec3, rd: wp.vec3, radius: float, half_height: float) -> tuple[int, wp.vec3]:
    pa = wp.vec3(0.0, 0.0, -half_height)
    pb = wp.vec3(0.0, 0.0, half_height)
    ra = radius
    rb = 0.0
    ba = pb - pa
    oa = ro - pa
    ob = ro - pb
    m0 = wp.dot(ba, ba)
    m1 = wp.dot(oa, ba)
    m2 = wp.dot(rd, ba)
    m3 = wp.dot(rd, oa)
    m5 = wp.dot(oa, oa)
    m9 = wp.dot(ob, ba)

    t_best = 1.0e20
    found = False

    if m1 < 0.0:
        if wp.dot(oa * m2 - rd * m1, oa * m2 - rd * m1) < (ra * ra * m2 * m2):
            t_base = -m1 / m2
            if wp.abs(t_base) < wp.abs(t_best):
                t_best = t_base
                found = True
    if m9 > 0.0:
        t_apex = -m9 / m2
        if wp.dot(ob + rd * t_apex, ob + rd * t_apex) < (rb * rb):
            if wp.abs(t_apex) < wp.abs(t_best):
                t_best = t_apex
                found = True

    rr = ra - rb
    hy = m0 + rr * rr
    k2 = m0 * m0 - m2 * m2 * hy
    k1 = m0 * m0 * m3 - m1 * m2 * hy + m0 * ra * (rr * m2 * 1.0)
    k0 = m0 * m0 * m5 - m1 * m1 * hy + m0 * ra * (rr * m1 * 2.0 - m0 * ra)
    h = k1 * k1 - k2 * k0
    if h >= 0.0:
        h_sqrt = wp.sqrt(h)
        for i in range(2):
            t = (-k1 - h_sqrt) / k2 if i == 0 else (-k1 + h_sqrt) / k2
            y = m1 + t * m2
            if y >= 0.0 and y <= m0:
                if wp.abs(t) < wp.abs(t_best):
                    t_best = t
                    found = True
    if not found:
        return int(IntersectionResult.NO_INTERSECTION), wp.vec3(0.0, 0.0, 0.0)
    return int(IntersectionResult.INTERSECTION), ro + t_best * rd


@wp.func
def ray_ellipsoid_intersection(ro: wp.vec3, rd: wp.vec3, radii: wp.vec3) -> tuple[int, wp.vec3]:
    ocn = wp.vec3(ro[0] / radii[0], ro[1] / radii[1], ro[2] / radii[2])
    rdn = wp.vec3(rd[0] / radii[0], rd[1] / radii[1], rd[2] / radii[2])
    a = wp.dot(rdn, rdn)
    b = wp.dot(ocn, rdn)
    c = wp.dot(ocn, ocn)
    h = b * b - a * (c - 1.0)
    if h < 0.0:
        return int(IntersectionResult.NO_INTERSECTION), wp.vec3(0.0, 0.0, 0.0)
    h = wp.sqrt(h)
    t1 = (-b - h) / a
    t2 = (-b + h) / a
    t = t1 if wp.abs(t1) < wp.abs(t2) else t2
    return int(IntersectionResult.INTERSECTION), ro + t * rd


@wp.func
def ray_surface_intersection_func(
    geom: GenericShapeData, rayOrigin: wp.vec3, rayDirection: wp.vec3, data_provider: SupportMapDataProvider
) -> tuple[int, wp.vec3]:
    ro = rayOrigin
    rd = rayDirection

    if geom.shape_type == int(GeoType.SPHERE):
        return ray_sphere_intersection(ro, rd, geom.scale[0])
    elif geom.shape_type == int(GeoType.CAPSULE):
        return ray_capsule_intersection(ro, rd, geom.scale[0], geom.scale[1])
    elif geom.shape_type == int(GeoType.CYLINDER):
        return ray_cylinder_intersection(ro, rd, geom.scale[0], geom.scale[1])
    elif geom.shape_type == int(GeoType.CONE):
        return ray_cone_intersection(ro, rd, geom.scale[0], geom.scale[1])
    elif geom.shape_type == int(GeoType.ELLIPSOID):
        return ray_ellipsoid_intersection(ro, rd, geom.scale)
    else:
        return int(IntersectionResult.NOT_SUPPORTED), wp.vec3(0.0, 0.0, 0.0)


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
                result = wp.vec3(
                    (a * a) * dir_safe[0] / denom, (b * b) * dir_safe[1] / denom, (c * c) * dir_safe[2] / denom
                )
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
