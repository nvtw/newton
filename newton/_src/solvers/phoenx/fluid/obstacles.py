# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.kernels import triangle_closest_point_barycentric
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import eval_shape_sdf
from newton._src.geometry.support_function import decode_vec3
from newton._src.geometry.types import GeoType
from newton._src.utils.heightfield import HeightfieldData, sample_sdf_heightfield


@wp.kernel(enable_backward=False)
def _shape_world_transforms(
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    world_transform: wp.array[wp.transform],
):
    shape = wp.tid()
    body = shape_body[shape]
    world_transform[shape] = shape_transform[shape]
    if body >= 0:
        world_transform[shape] = wp.transform_multiply(body_q[body], shape_transform[shape])


@wp.func
def _tetrahedron_contains(point: wp.vec3, scale: wp.vec3, encoded_d: wp.uint64):
    b = wp.vec3(0.0, 0.0, scale[0])
    c = wp.vec3(0.0, scale[1], scale[2])
    d = decode_vec3(encoded_d)
    basis = wp.mat33(b[0], c[0], d[0], b[1], c[1], d[1], b[2], c[2], d[2])
    bary = wp.inverse(basis) * point
    return bary[0] >= 0.0 and bary[1] >= 0.0 and bary[2] >= 0.0 and bary[0] + bary[1] + bary[2] <= 1.0


@wp.kernel(enable_backward=False)
def _rasterize(
    points: wp.array[wp.float32],
    world_transform: wp.array[wp.transform],
    shape_type: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source: wp.array[wp.uint64],
    shape_flags: wp.array[wp.int32],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_count: int,
    order: int,
    resolution: wp.vec3i,
    origin: wp.vec3,
    spacing: wp.vec3,
    smoothing: float,
    volume_fraction: wp.array[wp.float16],
):
    i, j, k, sp = wp.tid()
    k2 = order * order
    sx = sp % order
    sy = (sp // order) % order
    sz = sp // k2
    local = 0.5 * wp.vec3(points[sx] + 1.0, points[sy] + 1.0, points[sz] + 1.0)
    x_world = origin + wp.cw_mul(spacing, wp.vec3(float(i), float(j), float(k)) + local)
    chi = wp.float32(0.0)
    for shape in range(shape_count):
        if shape_flags[shape] & int(ShapeFlags.COLLIDE_SHAPES):
            geo = shape_type[shape]
            x_local = wp.transform_point(wp.transform_inverse(world_transform[shape]), x_world)
            scale = shape_scale[shape]
            phi = wp.float32(1.0e6)
            sdf_index = shape_sdf_index[shape]
            if geo == GeoType.HFIELD:
                hfield_index = shape_heightfield_index[shape]
                phi = sample_sdf_heightfield(heightfield_data[hfield_index], heightfield_elevations, x_local)
            elif geo == GeoType.TRIANGLE:
                a = wp.vec3(0.0)
                b = wp.vec3(0.0, 0.0, scale[0])
                c = wp.vec3(0.0, scale[1], scale[2])
                bary = triangle_closest_point_barycentric(a, b, c, x_local)
                closest = bary[0] * a + bary[1] * b + bary[2] * c
                phi = wp.length(x_local - closest) - wp.max(smoothing, 0.5 * wp.min(spacing) / float(order))
            elif geo == GeoType.TETRAHEDRON:
                if _tetrahedron_contains(x_local, scale, shape_source[shape]):
                    phi = -1.0
            elif geo == GeoType.PLANE:
                _plane_lower, phi, _plane_grad = eval_shape_sdf(geo, scale, x_local, sdf_index, texture_sdf_table)
                if scale[0] > 0.0 and scale[1] > 0.0:
                    phi -= wp.max(smoothing, 0.5 * wp.min(spacing) / float(order))
            elif geo == GeoType.MESH or geo == GeoType.CONVEX_MESH:
                if sdf_index >= 0:
                    _phi_lower, phi, _phi_grad = eval_shape_sdf(geo, scale, x_local, sdf_index, texture_sdf_table)
                else:
                    normalized = wp.cw_div(x_local, scale)
                    query = wp.mesh_query_point_sign_normal(shape_source[shape], normalized, 1.0e6)
                    if query.result:
                        closest = wp.mesh_eval_position(shape_source[shape], query.face, query.u, query.v)
                        phi = query.sign * wp.length(normalized - closest) * wp.min(wp.abs(scale))
            elif geo != GeoType.GAUSSIAN and geo != GeoType.NONE:
                _phi_lower, phi, _phi_grad = eval_shape_sdf(geo, scale, x_local, sdf_index, texture_sdf_table)
            shape_chi = wp.float32(0.0)
            if smoothing > 0.0:
                shape_chi = 0.5 * (1.0 - wp.tanh(phi / smoothing))
            elif phi < 0.0:
                shape_chi = 1.0
            chi = wp.max(chi, shape_chi)
    index = ((i * resolution[1] + j) * resolution[2] + k) * order * order * order + sp
    volume_fraction[index] = wp.float16(chi)


def rasterize_obstacles(solver, model, state=None, *, origin=(0.0, 0.0, 0.0), smoothing: float | None = None):
    """Snapshot Newton collision shapes into the fluid obstacle field."""
    if state is None:
        state = model.state()
    device = solver.device
    if model.device != device:
        raise ValueError("fluid solver and Newton model must use the same device")
    world_transform = wp.empty(model.shape_count, dtype=wp.transform, device=device)
    wp.launch(
        _shape_world_transforms,
        dim=model.shape_count,
        inputs=[state.body_q, model.shape_body, model.shape_transform, world_transform],
        device=device,
    )
    resolution = wp.vec3i(*solver.config.resolution)
    spacing = wp.vec3(*(solver.config.size[axis] / solver.config.resolution[axis] for axis in range(3)))
    if smoothing is None:
        smoothing = min(spacing) / solver.config.order
    points = wp.array(solver.points, device=device)
    wp.launch(
        _rasterize,
        dim=(*solver.config.resolution, solver.config.order**3),
        inputs=[
            points,
            world_transform,
            model.shape_type,
            model.shape_scale,
            model.shape_source_ptr,
            model.shape_flags,
            model._shape_sdf_index,
            model._texture_sdf_data,
            model.shape_heightfield_index,
            model.heightfield_data,
            model.heightfield_elevations,
            model.shape_count,
            solver.config.order,
            resolution,
            wp.vec3(*origin),
            spacing,
            smoothing,
            solver.volume_fraction,
        ],
        device=device,
    )
    return solver.volume_fraction
