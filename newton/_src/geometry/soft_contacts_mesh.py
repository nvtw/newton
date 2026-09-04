# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""BVH-culled deformable face contacts for rigid triangle meshes."""

import warp as wp

from .flags import ShapeFlags
from .sdf_texture import TextureSDFData
from .soft_contacts_sdf import (
    SDF_FACE_ITERS,
    SDF_LS_ITERS,
    _emit_soft_ef_contact,
    _shape_frames,
    eval_shape_sdf,
    optimize_face_sdf,
)


@wp.kernel
def create_soft_mesh_face_contacts(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_source: wp.array[wp.uint64],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    """Cull mesh-face pairs with the rigid triangle BVH before running SDF minimization."""
    tid = wp.tid()
    pair = face_pairs[tid]
    tri = pair[0]
    shape = pair[1]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    a_idx = tri_indices[tri, 0]
    b_idx = tri_indices[tri, 1]
    c_idx = tri_indices[tri, 2]
    radius = wp.max(particle_radius[a_idx], wp.max(particle_radius[b_idx], particle_radius[c_idx]))
    shape_contact_margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + shape_contact_margin + radius

    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    a = wp.transform_point(X_sw, particle_q[a_idx])
    b = wp.transform_point(X_sw, particle_q[b_idx])
    c = wp.transform_point(X_sw, particle_q[c_idx])

    # Most Cartesian face/shape pairs are nowhere near the rigid mesh. Reject them against its
    # scaled local AABB without paying for a texture lookup or BVH traversal.
    tri_lower = wp.min(a, wp.min(b, c)) - wp.vec3(threshold)
    tri_upper = wp.max(a, wp.max(b, c)) + wp.vec3(threshold)
    mesh_lower = shape_aabb_lower[shape]
    mesh_upper = shape_aabb_upper[shape]
    if (
        tri_upper[0] < mesh_lower[0]
        or tri_upper[1] < mesh_lower[1]
        or tri_upper[2] < mesh_lower[2]
        or tri_lower[0] > mesh_upper[0]
        or tri_lower[1] > mesh_upper[1]
        or tri_lower[2] > mesh_upper[2]
    ):
        return

    scale = shape_scale[shape]
    geo = shape_type[shape]
    inv_scale = wp.cw_div(wp.vec3(1.0), scale)
    a_mesh = wp.cw_mul(a, inv_scale)
    b_mesh = wp.cw_mul(b, inv_scale)
    c_mesh = wp.cw_mul(c, inv_scale)
    expansion = threshold * wp.abs(inv_scale)
    query = wp.mesh_query_aabb(
        shape_source[shape],
        wp.min(a_mesh, wp.min(b_mesh, c_mesh)) - expansion,
        wp.max(a_mesh, wp.max(b_mesh, c_mesh)) + expansion,
    )
    mesh_tri = wp.int32(0)
    surface_near = wp.mesh_query_aabb_next(query, mesh_tri)

    sdf_idx = shape_sdf_index[shape]
    if not surface_near:
        # A face wholly inside a closed mesh need not overlap its surface BVH. Preserve the previous
        # penetrating-face contact in that case; any face entering from outside crosses the surface
        # and therefore has a BVH hit above.
        centroid = (a + b + c) / 3.0
        _phi_lower, phi_centroid, _grad = eval_shape_sdf(geo, scale, centroid, sdf_idx, texture_sdf_table)
        if phi_centroid >= 0.0:
            return

    bary, x, phi, grad = optimize_face_sdf(
        geo,
        scale,
        a,
        b,
        c,
        sdf_idx,
        texture_sdf_table,
        SDF_FACE_ITERS,
        SDF_LS_ITERS,
    )
    if phi < threshold:
        y = x - phi * grad
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(a_idx, b_idx, c_idx),
            bary,
            shape,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0),
            wp.transform_vector(X_ws, grad),
        )


def launch_soft_mesh_face_contacts(*, model, state, contacts, margin: float, device, face_pairs, tid_base: int):
    """Launch BVH-culled SDF face minimization for rigid mesh shapes."""
    if len(face_pairs) == 0:
        return
    wp.launch(
        create_soft_mesh_face_contacts,
        dim=len(face_pairs),
        inputs=[
            face_pairs,
            state.particle_q,
            model.particle_radius,
            model.tri_indices,
            model.shape_body,
            model.shape_type,
            model.shape_flags,
            model.shape_transform,
            model.shape_scale,
            state.body_q,
            model.shape_source_ptr,
            model._shape_sdf_index,
            model._texture_sdf_data,
            model.shape_margin,
            model.shape_collision_aabb_lower,
            model.shape_collision_aabb_upper,
            margin,
            tid_base,
            contacts.soft_contact_max,
        ],
        outputs=[
            contacts.soft_contact_count,
            contacts.soft_contact_tids,
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ],
        device=device,
        # Texture-SDF face optimization is register-heavy; two warps per block balance occupancy and throughput.
        block_dim=64,
    )
