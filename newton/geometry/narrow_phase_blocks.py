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


@wp.struct
class Geom:
    transform: wp.transform
    scale: wp.vec3


@wp.struct
class ContactPoint:
    pos: wp.vec3
    normal: wp.vec3
    tangent: wp.vec3
    dist: float


@wp.func
def orthogonals(a: wp.vec3):
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if wp.length(a) == 0.0:
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)

    return b, c


@wp.func
def make_contact_frame(a: wp.vec3):
    a = wp.normalize(a)
    b, c = orthogonals(a)
    return a, b  # z (normal), x (tangent)


def get_plane_normal(transform: wp.transform):
    rot = wp.quat_to_matrix(transform.q)
    return wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])  # plane


@wp.func
def _plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float):
    dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
    pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
    return dist, pos


@wp.func
def plane_sphere(
    plane: Geom,
    sphere: Geom,
) -> ContactPoint:
    plane_normal = get_plane_normal(plane.transform)
    dist, pos = _plane_sphere(plane_normal, plane.transform.p, sphere.transform.p, sphere.scale[0])

    # Return contact frame using make_frame helper
    normal, tangent = make_contact_frame(plane_normal)
    return ContactPoint(pos, normal, tangent, dist)
