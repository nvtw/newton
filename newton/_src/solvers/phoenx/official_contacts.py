# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Normalize Newton soft contacts for the PhoenX PGS contact solver."""

from __future__ import annotations

import warp as wp

from newton._src.geometry.kernels import triangle_closest_point_barycentric
from newton._src.geometry.types import GeoType
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_set_friction,
    contact_set_friction_dynamic,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_side0_bary,
    cc_set_side1_bary,
)
from newton._src.solvers.phoenx.contact_endpoints import (
    SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE,
    SHAPE_ENDPOINT_KIND_RIGID,
    ShapeEndpoint,
)


@wp.kernel(enable_backward=False)
def override_soft_contact_column_friction_kernel(
    pair_source_idx: wp.array[wp.int32],
    pair_shape_a: wp.array[wp.int32],
    pair_shape_b: wp.array[wp.int32],
    num_contact_columns: wp.array[wp.int32],
    rigid_shape_count: wp.int32,
    soft_friction: wp.float32,
    contact_columns: ContactColumnContainer,
):
    """Apply Newton's absolute soft-contact friction to deformable columns."""

    column = wp.tid()
    if column >= num_contact_columns[0]:
        return
    pair = pair_source_idx[column]
    if pair_shape_a[pair] >= rigid_shape_count or pair_shape_b[pair] >= rigid_shape_count:
        contact_set_friction(contact_columns, column, soft_friction)
        contact_set_friction_dynamic(contact_columns, column, soft_friction)


@wp.kernel(enable_backward=False)
def pack_rigid_contact_prefix_kernel(
    source_count: wp.array[wp.int32],
    source_point0: wp.array[wp.vec3f],
    source_point1: wp.array[wp.vec3f],
    source_normal: wp.array[wp.vec3f],
    source_shape0: wp.array[wp.int32],
    source_shape1: wp.array[wp.int32],
    source_match: wp.array[wp.int32],
    source_margin0: wp.array[wp.float32],
    source_margin1: wp.array[wp.float32],
    destination_point0: wp.array[wp.vec3f],
    destination_point1: wp.array[wp.vec3f],
    destination_normal: wp.array[wp.vec3f],
    destination_shape0: wp.array[wp.int32],
    destination_shape1: wp.array[wp.int32],
    destination_match: wp.array[wp.int32],
    destination_margin0: wp.array[wp.float32],
    destination_margin1: wp.array[wp.float32],
):
    k = wp.tid()
    count = wp.min(source_count[0], source_point0.shape[0])
    if k >= count:
        return
    destination_point0[k] = source_point0[k]
    destination_point1[k] = source_point1[k]
    destination_normal[k] = source_normal[k]
    destination_shape0[k] = source_shape0[k]
    destination_shape1[k] = source_shape1[k]
    destination_match[k] = source_match[k]
    destination_margin0[k] = source_margin0[k]
    destination_margin1[k] = source_margin1[k]


@wp.kernel(enable_backward=False)
def pack_soft_contact_suffix_kernel(
    rigid_count: wp.array[wp.int32],
    soft_count: wp.array[wp.int32],
    soft_indices: wp.array[wp.vec3i],
    soft_barycentric: wp.array[wp.vec3f],
    soft_shape: wp.array[wp.int32],
    soft_body_pos: wp.array[wp.vec3f],
    soft_normal: wp.array[wp.vec3f],
    particle_position: wp.array[wp.vec3f],
    particle_radius: wp.array[wp.float32],
    shape_margin: wp.array[wp.float32],
    shape_count: wp.int32,
    num_bodies: wp.int32,
    destination_capacity: wp.int32,
    destination_count: wp.array[wp.int32],
    destination_point0: wp.array[wp.vec3f],
    destination_point1: wp.array[wp.vec3f],
    destination_normal: wp.array[wp.vec3f],
    destination_shape0: wp.array[wp.int32],
    destination_shape1: wp.array[wp.int32],
    destination_match: wp.array[wp.int32],
    destination_margin0: wp.array[wp.float32],
    destination_margin1: wp.array[wp.float32],
    destination_stiffness: wp.array[wp.float32],
    destination_damping: wp.array[wp.float32],
    destination_friction: wp.array[wp.float32],
    destination_endpoints: wp.array[ShapeEndpoint],
    destination_shape_body: wp.array[wp.int32],
    destination_shape_type: wp.array[wp.int32],
    destination_bary0: wp.array[wp.vec3f],
    destination_bary1: wp.array[wp.vec3f],
):
    j = wp.tid()
    nr = wp.min(rigid_count[0], destination_capacity)
    ns = wp.min(soft_count[0], soft_indices.shape[0])
    if j == 0:
        destination_count[0] = wp.min(destination_capacity, nr + ns)
    k = nr + j
    if j >= ns or k >= destination_capacity:
        return

    indices = soft_indices[j]
    bary = soft_barycentric[j]
    p0 = indices[0]
    p1 = indices[1]
    p2 = indices[2]
    if p0 < 0:
        return
    if p1 < 0:
        p1 = p0
    if p2 < 0:
        p2 = p0

    soft_point = bary[0] * particle_position[p0]
    radius = particle_radius[p0]
    if indices[1] >= 0:
        soft_point = soft_point + bary[1] * particle_position[indices[1]]
        radius = wp.max(radius, particle_radius[indices[1]])
    if indices[2] >= 0:
        soft_point = soft_point + bary[2] * particle_position[indices[2]]
        radius = wp.max(radius, particle_radius[indices[2]])

    shape = soft_shape[j]
    pseudo_shape = shape_count + j
    destination_point0[k] = soft_body_pos[j]
    destination_point1[k] = soft_point
    destination_normal[k] = soft_normal[j]
    destination_shape0[k] = shape
    destination_shape1[k] = pseudo_shape
    destination_match[k] = wp.int32(-1)
    destination_margin0[k] = shape_margin[shape]
    destination_margin1[k] = radius
    destination_stiffness[k] = wp.float32(0.0)
    destination_damping[k] = wp.float32(0.0)
    destination_friction[k] = wp.float32(1.0)
    destination_bary0[k] = wp.vec3f(0.0)
    destination_bary1[k] = bary

    endpoint = ShapeEndpoint()
    endpoint.kind = wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)
    endpoint.nodes = wp.vec4i(num_bodies + p0, num_bodies + p1, num_bodies + p2, wp.int32(-1))
    destination_endpoints[pseudo_shape] = endpoint
    destination_shape_body[pseudo_shape] = num_bodies + p0
    destination_shape_type[pseudo_shape] = wp.int32(GeoType.TRIANGLE)


@wp.kernel(enable_backward=False)
def pack_self_vertex_triangle_kernel(
    collision_pairs: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    particle_position: wp.array[wp.vec3f],
    contact_margin: wp.float32,
    contact_gap: wp.float32,
    shape_base: wp.int32,
    num_bodies: wp.int32,
    destination_capacity: wp.int32,
    destination_count: wp.array[wp.int32],
    destination_point0: wp.array[wp.vec3f],
    destination_point1: wp.array[wp.vec3f],
    destination_normal: wp.array[wp.vec3f],
    destination_shape0: wp.array[wp.int32],
    destination_shape1: wp.array[wp.int32],
    destination_match: wp.array[wp.int32],
    destination_margin0: wp.array[wp.float32],
    destination_margin1: wp.array[wp.float32],
    destination_stiffness: wp.array[wp.float32],
    destination_damping: wp.array[wp.float32],
    destination_friction: wp.array[wp.float32],
    destination_endpoints: wp.array[ShapeEndpoint],
    destination_shape_body: wp.array[wp.int32],
    destination_shape_type: wp.array[wp.int32],
    destination_bary0: wp.array[wp.vec3f],
    destination_bary1: wp.array[wp.vec3f],
):
    pair = wp.tid()
    vertex = collision_pairs[2 * pair]
    triangle = collision_pairs[2 * pair + 1]
    if vertex < 0 or triangle < 0:
        return
    ta = tri_indices[triangle, 0]
    tb = tri_indices[triangle, 1]
    tc = tri_indices[triangle, 2]
    x = particle_position[vertex]
    a = particle_position[ta]
    b = particle_position[tb]
    c = particle_position[tc]
    bary = triangle_closest_point_barycentric(a, b, c, x)
    q = bary[0] * a + bary[1] * b + bary[2] * c
    delta = q - x
    distance = wp.length(delta)
    if distance > contact_margin + contact_gap:
        return
    normal = wp.vec3f(0.0, 1.0, 0.0)
    if distance > wp.float32(1.0e-8):
        normal = delta / distance
    else:
        face_normal = wp.cross(b - a, c - a)
        face_length = wp.length(face_normal)
        if face_length > wp.float32(1.0e-8):
            normal = face_normal / face_length

    k = wp.atomic_add(destination_count, 0, 1)
    if k >= destination_capacity:
        return
    shape0 = shape_base + 2 * k
    shape1 = shape0 + 1
    destination_point0[k] = x
    destination_point1[k] = q
    destination_normal[k] = normal
    destination_shape0[k] = shape0
    destination_shape1[k] = shape1
    destination_match[k] = wp.int32(-1)
    destination_margin0[k] = wp.float32(0.0)
    destination_margin1[k] = contact_margin
    destination_stiffness[k] = wp.float32(0.0)
    destination_damping[k] = wp.float32(0.0)
    destination_friction[k] = wp.float32(1.0)
    destination_bary0[k] = wp.vec3f(1.0, 0.0, 0.0)
    destination_bary1[k] = bary

    ep0 = ShapeEndpoint()
    ep0.kind = wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)
    node0 = num_bodies + vertex
    ep0.nodes = wp.vec4i(node0, node0, node0, wp.int32(-1))
    ep1 = ShapeEndpoint()
    ep1.kind = wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)
    ep1.nodes = wp.vec4i(num_bodies + ta, num_bodies + tb, num_bodies + tc, wp.int32(-1))
    destination_endpoints[shape0] = ep0
    destination_endpoints[shape1] = ep1
    destination_shape_body[shape0] = node0
    destination_shape_body[shape1] = num_bodies + ta
    destination_shape_type[shape0] = wp.int32(GeoType.TRIANGLE)
    destination_shape_type[shape1] = wp.int32(GeoType.TRIANGLE)


@wp.kernel(enable_backward=False)
def pack_self_edge_edge_kernel(
    collision_pairs: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    particle_position: wp.array[wp.vec3f],
    contact_margin: wp.float32,
    contact_gap: wp.float32,
    parallel_epsilon: wp.float32,
    shape_base: wp.int32,
    num_bodies: wp.int32,
    destination_capacity: wp.int32,
    destination_count: wp.array[wp.int32],
    destination_point0: wp.array[wp.vec3f],
    destination_point1: wp.array[wp.vec3f],
    destination_normal: wp.array[wp.vec3f],
    destination_shape0: wp.array[wp.int32],
    destination_shape1: wp.array[wp.int32],
    destination_match: wp.array[wp.int32],
    destination_margin0: wp.array[wp.float32],
    destination_margin1: wp.array[wp.float32],
    destination_stiffness: wp.array[wp.float32],
    destination_damping: wp.array[wp.float32],
    destination_friction: wp.array[wp.float32],
    destination_endpoints: wp.array[ShapeEndpoint],
    destination_shape_body: wp.array[wp.int32],
    destination_shape_type: wp.array[wp.int32],
    destination_bary0: wp.array[wp.vec3f],
    destination_bary1: wp.array[wp.vec3f],
):
    pair = wp.tid()
    edge0 = collision_pairs[2 * pair]
    edge1 = collision_pairs[2 * pair + 1]
    if edge0 < 0 or edge1 < 0 or edge0 >= edge1:
        return
    a0i = edge_indices[edge0, 2]
    a1i = edge_indices[edge0, 3]
    b0i = edge_indices[edge1, 2]
    b1i = edge_indices[edge1, 3]
    a0 = particle_position[a0i]
    a1 = particle_position[a1i]
    b0 = particle_position[b0i]
    b1 = particle_position[b1i]
    st = wp.closest_point_edge_edge(a0, a1, b0, b1, parallel_epsilon)
    u = st[0]
    v = st[1]
    p = a0 + u * (a1 - a0)
    q = b0 + v * (b1 - b0)
    delta = q - p
    distance = wp.length(delta)
    if distance > contact_margin + contact_gap or distance <= wp.float32(1.0e-8):
        return
    normal = delta / distance

    k = wp.atomic_add(destination_count, 0, 1)
    if k >= destination_capacity:
        return
    shape0 = shape_base + 2 * k
    shape1 = shape0 + 1
    destination_point0[k] = p
    destination_point1[k] = q
    destination_normal[k] = normal
    destination_shape0[k] = shape0
    destination_shape1[k] = shape1
    destination_match[k] = wp.int32(-1)
    destination_margin0[k] = wp.float32(0.0)
    destination_margin1[k] = contact_margin
    destination_stiffness[k] = wp.float32(0.0)
    destination_damping[k] = wp.float32(0.0)
    destination_friction[k] = wp.float32(1.0)
    destination_bary0[k] = wp.vec3f(1.0 - u, u, 0.0)
    destination_bary1[k] = wp.vec3f(1.0 - v, v, 0.0)

    ep0 = ShapeEndpoint()
    ep0.kind = wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)
    ep0.nodes = wp.vec4i(num_bodies + a0i, num_bodies + a1i, num_bodies + a0i, wp.int32(-1))
    ep1 = ShapeEndpoint()
    ep1.kind = wp.int32(SHAPE_ENDPOINT_KIND_CLOTH_TRIANGLE)
    ep1.nodes = wp.vec4i(num_bodies + b0i, num_bodies + b1i, num_bodies + b0i, wp.int32(-1))
    destination_endpoints[shape0] = ep0
    destination_endpoints[shape1] = ep1
    destination_shape_body[shape0] = num_bodies + a0i
    destination_shape_body[shape1] = num_bodies + b0i
    destination_shape_type[shape0] = wp.int32(GeoType.TRIANGLE)
    destination_shape_type[shape1] = wp.int32(GeoType.TRIANGLE)


@wp.kernel(enable_backward=False)
def overwrite_contact_barycentric_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    endpoints: wp.array[ShapeEndpoint],
    bary0: wp.array[wp.vec3f],
    bary1: wp.array[wp.vec3f],
    capacity: wp.int32,
    contact_container: ContactContainer,
):
    k = wp.tid()
    count = wp.min(contact_count[0], capacity)
    if k >= count:
        return
    endpoint0 = endpoints[contact_shape0[k]]
    endpoint1 = endpoints[contact_shape1[k]]
    if endpoint0.kind != wp.int32(SHAPE_ENDPOINT_KIND_RIGID):
        cc_set_side0_bary(contact_container, k, bary0[k])
    if endpoint1.kind != wp.int32(SHAPE_ENDPOINT_KIND_RIGID):
        cc_set_side1_bary(contact_container, k, bary1[k])
