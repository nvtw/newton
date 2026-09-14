# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


"""Full-surface mesh contacts with local feature validity and final-slot emission.

Vertex-face, face-vertex, and edge-edge pairs describe the same surface-distance
problem at every deformable resolution. Incident-feature cones reject redundant
representations without discarding distinct nearby surface patches. Detection
records immutable feature IDs; a separate differentiable pass evaluates only
the accepted final contacts. No intermediate contact pool is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .collision_core import transform_normal_with_scale
from .flags import ParticleFlags, ShapeFlags
from .kernels import triangle_closest_point
from .soft_contacts_sdf import _shape_frames
from .tri_mesh_collision import TriMeshCollisionDetector
from .types import GeoType

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State
    from .types import Mesh

_MESH_QUERY_LANES = wp.constant(8)


@wp.func
def _cone_valid(
    mesh: wp.uint64, scale: wp.vec3, point: wp.vec3, diff: wp.vec3, span: wp.vec3i, neighbors: wp.array[int]
):
    # Outward separation must not point into any incident feature. The allowance
    # bounds position-subtraction roundoff, rather than imposing a world-unit gap.
    valid = bool(True)
    for k in range(span[0], span[1]):
        neighbor = wp.cw_mul(wp.mesh_get_point(mesh, neighbors[k]), scale)
        direction = neighbor - point
        tolerance = (
            2.0e-6
            * (wp.length(point) + wp.length(neighbor) + wp.length(diff))
            * (wp.length(direction) + wp.length(diff))
        )
        if wp.dot(diff, direction) > tolerance:
            valid = False
    return valid


@wp.func
def _face_valid(
    mesh: wp.uint64,
    scale: wp.vec3,
    point: wp.vec3,
    diff: wp.vec3,
    bary: wp.vec3,
    face: int,
    offset: int,
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    neighbors: wp.array[int],
):
    count = int(0)
    positive = int(0)
    zero = int(0)
    for k in range(3):
        if bary[k] > 0.0:
            count += 1
            positive = k
        else:
            zero = k
    if count == 1:
        span = vertex_spans[offset + 3 * face + positive]
        return span[2] == face and _cone_valid(mesh, scale, point, diff, span, neighbors)
    if count == 2:
        span = edge_spans[offset + 3 * face + zero]
        return span[2] == face and _cone_valid(mesh, scale, point, diff, span, neighbors)
    return True


def _build_feature_adjacency(model: Model, vertex_table: wp.array, edge_table: wp.array):
    """Build fixed incident-feature spans and canonical face ownership."""
    vt, et = vertex_table.numpy(), edge_table.numpy()
    offsets = np.zeros(model.shape_count, dtype=np.int32)
    vertex_spans, edge_spans, neighbors = [], [], []
    tv = np.zeros((len(vt), 3), dtype=np.int32)
    ee = np.zeros((len(et), 3), dtype=np.int32)
    for shape in range(model.shape_count):
        mesh = model.shape_source[shape]
        if mesh is None or not hasattr(mesh, "indices"):
            continue
        idx = np.asarray(mesh.indices).reshape(-1)
        canon = mesh._canonical_vertex_ids()[idx]
        representative = {}
        incident = {}
        opposite = {}
        edge_owner = {}
        for slot, vertex in enumerate(canon):
            representative.setdefault(int(vertex), slot)
            incident.setdefault(int(vertex), set())
        for face, tri in enumerate(canon.reshape(-1, 3)):
            for k in range(3):
                v = int(tri[k])
                a, b = int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
                incident[v].update((a, b))
                key = tuple(sorted((a, b)))
                opposite.setdefault(key, set()).add(v)
                edge_owner.setdefault(key, face)

        def span(vertices, owner, representatives=representative):
            start = len(neighbors)
            neighbors.extend(representatives[v] for v in sorted(vertices))
            return (start, len(neighbors), owner)

        vs = {v: span(n, representative[v] // 3) for v, n in incident.items()}
        es = {key: span(n, edge_owner[key]) for key, n in opposite.items()}
        offsets[shape] = len(vertex_spans)
        for slot, v in enumerate(canon):
            vertex_spans.append(vs[int(v)])
            base, k = (slot // 3) * 3, slot % 3
            key = tuple(sorted((int(canon[base + (k + 1) % 3]), int(canon[base + (k + 2) % 3]))))
            edge_spans.append(es[key])
        for row in np.flatnonzero(vt[:, 0] == shape):
            tv[row] = vs[int(canon[vt[row, 1]])]
        for row in np.flatnonzero(et[:, 0] == shape):
            key = tuple(sorted((int(canon[et[row, 1]]), int(canon[et[row, 2]]))))
            ee[row] = es[key]
    if max(len(vertex_spans), len(edge_spans), len(neighbors)) > np.iinfo(np.int32).max:
        raise ValueError("Mesh contact topology exceeds 32-bit indexing capacity.")
    arrays = [wp.array(offsets, dtype=int, device=model.device)]
    arrays.extend(
        wp.array(np.asarray(x, dtype=np.int32).reshape(-1, 3), dtype=wp.vec3i, device=model.device)
        for x in (vertex_spans, edge_spans, tv, ee)
    )
    arrays.append(wp.array(neighbors, dtype=int, device=model.device))
    return arrays


CONTACT_NORMAL_DEGENERATE_EPS = wp.constant(1.0e-6)


def _mesh_feature_data(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique face-vertex slots, edges, and outward reference normals."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    idx = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if idx.size == 0 or verts.size == 0:
        empty_i = np.empty(0, dtype=np.int32)
        return empty_i, np.empty((0, 3), np.float32), np.empty((0, 2), np.int32), np.empty((0, 3), np.float32)
    tris = idx.reshape(-1, 3)
    canonical = mesh._canonical_vertex_ids()
    n_canon = int(canonical.max()) + 1

    orig_edges, slot_keys, _sort_order, _keys_sorted, face_normals, face_norms = mesh._build_edge_slot_topology()
    valid_faces = face_norms > 0.0
    fn_unit = np.zeros_like(face_normals)
    fn_unit[valid_faces] = face_normals[valid_faces] / face_norms[valid_faces, None]

    def _unit(rows: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(rows, axis=1)
        out = np.zeros_like(rows)
        nz = lengths > 0.0
        out[nz] = rows[nz] / lengths[nz, None]
        out[~nz] = (0.0, 0.0, 1.0)  # fully degenerate feature: any fixed unit direction
        return out

    vertex_normal_accumulation = np.zeros((n_canon, 3), dtype=np.float64)
    corner_edges = ((1, 2), (2, 0), (0, 1))  # corner k spans the edges to the other two corners
    for k, (a, b) in enumerate(corner_edges):
        da = verts[tris[:, a]] - verts[tris[:, k]]
        db = verts[tris[:, b]] - verts[tris[:, k]]
        la = np.linalg.norm(da, axis=1)
        lb = np.linalg.norm(db, axis=1)
        valid_corners = valid_faces & (la > 0.0) & (lb > 0.0)
        cos_angle = np.zeros(len(tris))
        cos_angle[valid_corners] = np.clip(
            np.einsum("ij,ij->i", da[valid_corners], db[valid_corners]) / (la[valid_corners] * lb[valid_corners]),
            -1.0,
            1.0,
        )
        angle = np.where(valid_corners, np.arccos(cos_angle), 0.0)
        np.add.at(vertex_normal_accumulation, canonical[tris[:, k]], fn_unit * angle[:, None])

    canon_per_face_vertex = canonical[idx]
    used_canon, first_index = np.unique(canon_per_face_vertex, return_index=True)
    vertex_table = first_index.astype(np.int32)
    vertex_normals = _unit(vertex_normal_accumulation[used_canon]).astype(np.float32)

    index_of_canon = np.full(n_canon, -1, dtype=np.int32)
    index_of_canon[used_canon] = vertex_table
    _, first_idx, inverse = np.unique(slot_keys, return_index=True, return_inverse=True)
    edge_normal_accumulation = np.zeros((len(first_idx), 3), dtype=np.float64)
    np.add.at(edge_normal_accumulation, inverse, np.repeat(fn_unit, 3, axis=0))
    edge_outward = _unit(edge_normal_accumulation).astype(np.float32)
    edge_canon = canonical[orig_edges[first_idx]]
    edge_table = np.column_stack((index_of_canon[edge_canon[:, 0]], index_of_canon[edge_canon[:, 1]])).astype(np.int32)

    return vertex_table, vertex_normals, edge_table, edge_outward


def _build_rigid_features(
    model: Model, bvh_shape_mask: np.ndarray
) -> tuple[wp.array[wp.vec2i], wp.array[wp.vec3], wp.array[wp.vec3i], wp.array[wp.vec3]]:

    device = model.device
    vertex_rows: list[np.ndarray] = []
    vertex_normals: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    edge_outwards: list[np.ndarray] = []
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for shape in np.flatnonzero(bvh_shape_mask):
        mesh = model.shape_source[shape]
        if mesh is None:
            raise ValueError(f"mesh/convex shape {int(shape)} has no shape_source Mesh")
        key = hash(mesh)
        data = cache.get(key)
        if data is None:
            data = _mesh_feature_data(mesh)
            cache[key] = data
        v_table, v_normals, e_table, e_outward = data
        if len(v_table):
            vertex_rows.append(np.column_stack((np.full(len(v_table), shape, np.int32), v_table)))
            vertex_normals.append(v_normals)
        if len(e_table):
            edge_rows.append(np.column_stack((np.full(len(e_table), shape, np.int32), e_table)))
            edge_outwards.append(e_outward)

    def _stack(rows: list[np.ndarray], width: int) -> np.ndarray:
        return np.concatenate(rows) if rows else np.empty((0, width), np.int32)

    def _stackf(rows: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(rows) if rows else np.empty((0, 3), np.float32)

    return (
        wp.array(_stack(vertex_rows, 2), dtype=wp.vec2i, device=device),
        wp.array(_stackf(vertex_normals), dtype=wp.vec3, device=device),
        wp.array(_stack(edge_rows, 3), dtype=wp.vec3i, device=device),
        wp.array(_stackf(edge_outwards), dtype=wp.vec3, device=device),
    )


_MESH_FEATURE_VT = wp.constant(0)
_MESH_FEATURE_TV = wp.constant(1)
_MESH_FEATURE_EE = wp.constant(2)


@wp.func
def _append_mesh_contact(
    family: wp.int32,
    soft_feature: wp.int32,
    rigid_shape: wp.int32,
    rigid_feature: wp.int32,
    contact_max: wp.int32,
    contact_count: wp.array[wp.int32],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    # Saturate at capacity + 1 instead of wrapping on a dense overlap.
    index = wp.atomic_add(contact_count, 0, 0)
    claimed = bool(False)
    while index <= contact_max:
        previous = wp.atomic_cas(contact_count, 0, index, index + 1)
        if previous == index:
            claimed = True
            break
        index = previous
    if claimed and index == contact_max:
        wp.printf("Mesh soft-contact capacity exceeded (%d); this collision result is incomplete.\n", contact_max)
    if claimed and index >= 0 and index < contact_max:
        features[index] = wp.vec3i(family, soft_feature, rigid_feature)
        contact_shapes[index] = rigid_shape


@wp.func
def _oriented_contact_normal(diff: wp.vec3, d: float, reference: wp.vec3):

    if d > CONTACT_NORMAL_DEGENERATE_EPS and wp.dot(diff, reference) >= 0.0:
        return diff / d
    return reference


@wp.func
def _face_vertex_velocity(mesh: wp.uint64, index: wp.int32):

    face = index / 3
    corner = index - face * 3
    u = wp.where(corner == 0, 1.0, 0.0)
    v = wp.where(corner == 1, 1.0, 0.0)
    return wp.mesh_eval_velocity(mesh, face, u, v)


@wp.func
def _feature_mesh_sign(mesh: wp.uint64, X_sw: wp.transform, scale: wp.vec3, soft: wp.vec3):
    query_point = wp.cw_div(wp.transform_point(X_sw, soft), scale)
    query = wp.mesh_query_point_sign_normal(mesh, query_point, 1.0e6)
    if not query.result:
        return int(0)
    surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    uncertainty = 2.0e-6 * (wp.length(query_point) + wp.length(surface))
    if wp.length(query_point - surface) <= uncertainty:
        # At a boundary, use the candidate feature normal to resolve the side.
        return int(2)
    return wp.where(query.sign < 0.0, -1, 1)


@wp.kernel(enable_backward=False)
def _detect_mesh_vertex_contacts(
    vt_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_margin: wp.array[float],
    gap: float,
    contact_max: wp.int32,
    face_offsets: wp.array[int],
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    tv_spans: wp.array[wp.vec3i],
    ee_spans: wp.array[wp.vec3i],
    neighbors: wp.array[int],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int32],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    tid = wp.tid() // _MESH_QUERY_LANES
    lane = wp.tid() % _MESH_QUERY_LANES
    pair = vt_pairs[tid]
    particle_index = pair[0]
    shape_index = pair[1]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    px = particle_q[particle_index]
    radius = particle_radius[particle_index]
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    x_local = wp.transform_point(X_sw, px)
    scale = shape_scale[shape_index]
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    threshold = gap + s_margin + radius

    mesh = shape_source_ptr[shape_index]
    min_scale = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
    x_mesh = wp.cw_div(x_local, scale)
    r_mesh = threshold / min_scale
    nearest = wp.mesh_query_point_sign_normal(mesh, x_mesh, r_mesh)
    if not nearest.result:
        return
    vertex_sign = wp.where(nearest.sign < 0.0, -1, 1)
    lower = wp.vec3(x_mesh[0] - r_mesh, x_mesh[1] - r_mesh, x_mesh[2] - r_mesh)
    upper = wp.vec3(x_mesh[0] + r_mesh, x_mesh[1] + r_mesh, x_mesh[2] + r_mesh)

    query = wp.mesh_query_aabb(mesh, lower, upper)
    face = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face):
        if face % _MESH_QUERY_LANES != lane:
            continue
        a = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 0), scale)
        b = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 1), scale)
        c = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 2), scale)
        cp, rigid_bary, _feature = triangle_closest_point(a, b, c, x_local)
        if wp.length(x_local - cp) < threshold:
            if wp.length_sq(wp.cross(b - a, c - a)) == 0.0:
                continue  # degenerate sliver: no meaningful normal, neighbors still report
            if not _face_valid(
                mesh,
                scale,
                cp,
                float(vertex_sign) * (x_local - cp),
                rigid_bary,
                face,
                face_offsets[shape_index],
                vertex_spans,
                edge_spans,
                neighbors,
            ):
                continue
            sign = vertex_sign
            if sign != 0:
                _append_mesh_contact(
                    _MESH_FEATURE_VT + wp.where(sign < 0, 8, 0),
                    particle_index,
                    shape_index,
                    face,
                    contact_max,
                    contact_count,
                    features,
                    contact_shapes,
                )


@wp.kernel(enable_backward=False)
def _detect_mesh_face_contacts(
    rigid_vertex_table: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[float],
    bvh_tris_id: wp.uint64,
    bvh_tris_group_roots: wp.array[wp.int32],
    world_count: wp.int32,
    gap: float,
    max_particle_radius: float,
    contact_max: wp.int32,
    face_offsets: wp.array[int],
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    tv_spans: wp.array[wp.vec3i],
    ee_spans: wp.array[wp.vec3i],
    neighbors: wp.array[int],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int32],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    tid = wp.tid() // _MESH_QUERY_LANES
    lane = wp.tid() % _MESH_QUERY_LANES
    entry = rigid_vertex_table[tid]
    shape_index = entry[0]
    index = entry[1]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    scale = shape_scale[shape_index]
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    x_local = wp.cw_mul(wp.mesh_get_point(mesh, index), scale)
    x_w = wp.transform_point(X_ws, x_local)
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    bound = gap + s_margin + max_particle_radius
    lower = wp.vec3(x_w[0] - bound, x_w[1] - bound, x_w[2] - bound)
    upper = wp.vec3(x_w[0] + bound, x_w[1] + bound, x_w[2] + bound)

    rigid_world = shape_world[shape_index]

    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)

        if rigid_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_tris_group_roots[rigid_world]
            else:
                group_root = bvh_tris_group_roots[world_count]
            run_query = group_root >= 0

        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_tris_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_tris_id, lower, upper, group_root)

            tri_index = wp.int32(0)
            while wp.bvh_query_next(query, tri_index):
                if tri_index % _MESH_QUERY_LANES != lane:
                    continue
                t0 = tri_indices[tri_index, 0]
                t1 = tri_indices[tri_index, 1]
                t2 = tri_indices[tri_index, 2]
                active = (
                    (particle_flags[t0] & ParticleFlags.ACTIVE)
                    | (particle_flags[t1] & ParticleFlags.ACTIVE)
                    | (particle_flags[t2] & ParticleFlags.ACTIVE)
                )
                if active == 0:
                    continue

                cp, bary, _feature = triangle_closest_point(particle_q[t0], particle_q[t1], particle_q[t2], x_w)
                r_soft = bary[0] * particle_radius[t0] + bary[1] * particle_radius[t1] + bary[2] * particle_radius[t2]
                if wp.length(cp - x_w) < gap + s_margin + r_soft:
                    if bary[0] == 1.0 or bary[1] == 1.0 or bary[2] == 1.0:
                        continue
                    cp_local = wp.transform_point(_X_sw, cp)
                    diff = cp_local - x_local
                    outward = _cone_valid(mesh, scale, x_local, diff, tv_spans[tid], neighbors)
                    inward = _cone_valid(mesh, scale, x_local, -diff, tv_spans[tid], neighbors)
                    if not outward and not inward:
                        continue
                    sign = _feature_mesh_sign(mesh, _X_sw, scale, cp)
                    if sign == 2:
                        reference = transform_normal_with_scale(X_ws, scale, vertex_outward[tid])
                        sign = wp.where(wp.dot(cp - x_w, reference) < 0.0, -1, 1)
                    if (sign > 0 and outward) or (sign < 0 and inward):
                        _append_mesh_contact(
                            _MESH_FEATURE_TV + wp.where(sign < 0, 8, 0),
                            tri_index,
                            shape_index,
                            tid,
                            contact_max,
                            contact_count,
                            features,
                            contact_shapes,
                        )


@wp.kernel(enable_backward=False)
def _detect_mesh_edge_contacts(
    rigid_edge_table: wp.array[wp.vec3i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_world: wp.array[wp.int32],
    shape_margin: wp.array[float],
    bvh_edges_id: wp.uint64,
    bvh_edges_group_roots: wp.array[wp.int32],
    world_count: wp.int32,
    edge_edge_parallel_epsilon: float,
    gap: float,
    max_particle_radius: float,
    contact_max: wp.int32,
    face_offsets: wp.array[int],
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    tv_spans: wp.array[wp.vec3i],
    ee_spans: wp.array[wp.vec3i],
    neighbors: wp.array[int],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int32],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    tid = wp.tid() // _MESH_QUERY_LANES
    lane = wp.tid() % _MESH_QUERY_LANES
    entry = rigid_edge_table[tid]
    shape_index = entry[0]
    index0 = entry[1]
    index1 = entry[2]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    scale = shape_scale[shape_index]
    _X_bs, X_ws, _X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    r0_w = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, index0), scale))
    r1_w = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, index1), scale))
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0
    bound = gap + s_margin + max_particle_radius
    lower = wp.min(r0_w, r1_w)
    upper = wp.max(r0_w, r1_w)
    lower = wp.vec3(lower[0] - bound, lower[1] - bound, lower[2] - bound)
    upper = wp.vec3(upper[0] + bound, upper[1] + bound, upper[2] + bound)

    rigid_world = shape_world[shape_index]

    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)

        if rigid_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_edges_group_roots[rigid_world]
            else:
                group_root = bvh_edges_group_roots[world_count]
            run_query = group_root >= 0

        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_edges_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_edges_id, lower, upper, group_root)

            edge_index = wp.int32(0)
            while wp.bvh_query_next(query, edge_index):
                if edge_index % _MESH_QUERY_LANES != lane:
                    continue
                sv0 = edge_indices[edge_index, 2]
                sv1 = edge_indices[edge_index, 3]
                active = (particle_flags[sv0] & ParticleFlags.ACTIVE) | (particle_flags[sv1] & ParticleFlags.ACTIVE)
                if active == 0:
                    continue

                std = wp.closest_point_edge_edge(
                    r0_w, r1_w, particle_q[sv0], particle_q[sv1], edge_edge_parallel_epsilon
                )
                r_soft = wp.max(particle_radius[sv0], particle_radius[sv1])
                if std[2] < gap + s_margin + r_soft:
                    soft_point = particle_q[sv0] + std[1] * (particle_q[sv1] - particle_q[sv0])
                    rigid_point = r0_w + std[0] * (r1_w - r0_w)
                    if std[0] <= 0.0 or std[0] >= 1.0 or std[1] <= 0.0 or std[1] >= 1.0:
                        continue
                    rigid_local = wp.transform_point(_X_sw, rigid_point)
                    diff_local = wp.transform_vector(_X_sw, soft_point - rigid_point)
                    outward = _cone_valid(mesh, scale, rigid_local, diff_local, ee_spans[tid], neighbors)
                    inward = _cone_valid(mesh, scale, rigid_local, -diff_local, ee_spans[tid], neighbors)
                    if not outward and not inward:
                        continue
                    sign = _feature_mesh_sign(mesh, _X_sw, scale, soft_point)
                    if sign == 2:
                        reference = transform_normal_with_scale(X_ws, scale, edge_outward[tid])
                        sign = wp.where(wp.dot(soft_point - rigid_point, reference) < 0.0, -1, 1)
                    if not ((sign > 0 and outward) or (sign < 0 and inward)):
                        continue
                    valid_soft = bool(True)
                    for side in range(2):
                        opposite = edge_indices[edge_index, side]
                        if opposite >= 0:
                            direction = particle_q[opposite] - soft_point
                            diff_soft = float(sign) * (rigid_point - soft_point)
                            if wp.dot(diff_soft, direction) > 2.0e-6 * (
                                wp.length(soft_point) + wp.length(rigid_point) + wp.length(particle_q[opposite])
                            ) * (wp.length(diff_soft) + wp.length(direction)):
                                valid_soft = False
                    if not valid_soft:
                        continue
                    if sign != 0:
                        _append_mesh_contact(
                            _MESH_FEATURE_EE + wp.where(sign < 0, 8, 0),
                            edge_index,
                            shape_index,
                            tid,
                            contact_max,
                            contact_count,
                            features,
                            contact_shapes,
                        )


@wp.kernel
def _evaluate_mesh_contacts(
    contact_count: wp.array[wp.int32],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
    contact_max: wp.int32,
    shape_type: wp.array[int],
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    rigid_vertex_table: wp.array[wp.vec2i],
    rigid_vertex_normals: wp.array[wp.vec3],
    rigid_edge_table: wp.array[wp.vec3i],
    rigid_edge_outward_dirs: wp.array[wp.vec3],
    edge_edge_parallel_epsilon: float,
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):

    tid = wp.tid()
    if tid >= wp.min(contact_count[0], contact_max):
        return
    shape = contact_shapes[tid]
    if shape_type[shape] != GeoType.MESH and shape_type[shape] != GeoType.CONVEX_MESH:
        return
    feature = features[tid]
    family = feature[0] & 7
    soft_feature = feature[1]
    if feature[0] < 0:
        return
    shape_index = contact_shapes[tid]
    rigid_feature = feature[2]

    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    scale = shape_scale[shape_index]

    particle = wp.int32(-1)
    corners = wp.vec3i(-1, -1, -1)
    bary = wp.vec3(0.0)
    body_pos = wp.vec3(0.0)
    body_vel = wp.vec3(0.0)
    normal = wp.vec3(0.0)

    if family == _MESH_FEATURE_VT:
        particle = soft_feature
        corners = wp.vec3i(particle, -1, -1)
        bary = wp.vec3(1.0, 0.0, 0.0)
        face = rigid_feature
        x_local = wp.transform_point(X_sw, particle_q[particle])
        a = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 0), scale)
        b = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 1), scale)
        c = wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 2), scale)
        cp, rigid_bary, _feature = triangle_closest_point(a, b, c, x_local)
        diff = x_local - cp
        det_sign = wp.sign(scale[0] * scale[1] * scale[2])
        tri_n = wp.normalize(wp.cross(b - a, c - a)) * det_sign
        normal = wp.transform_vector(X_ws, _oriented_contact_normal(diff, wp.length(diff), tri_n))
        v_local = wp.cw_mul(wp.mesh_eval_velocity(mesh, face, rigid_bary[0], rigid_bary[1]), scale)
        body_pos = wp.transform_point(X_bs, cp)
        body_vel = wp.transform_vector(X_bs, v_local)
    elif family == _MESH_FEATURE_TV:
        vertex_entry = rigid_vertex_table[rigid_feature]
        index = vertex_entry[1]
        t0 = tri_indices[soft_feature, 0]
        t1 = tri_indices[soft_feature, 1]
        t2 = tri_indices[soft_feature, 2]
        corners = wp.vec3i(t0, t1, t2)
        x_local = wp.cw_mul(wp.mesh_get_point(mesh, index), scale)
        x_w = wp.transform_point(X_ws, x_local)
        cp, bary, _feature = triangle_closest_point(particle_q[t0], particle_q[t1], particle_q[t2], x_w)
        diff = cp - x_w
        n_ref = transform_normal_with_scale(X_ws, scale, rigid_vertex_normals[rigid_feature])
        normal = _oriented_contact_normal(diff, wp.length(diff), n_ref)
        v_local = wp.cw_mul(_face_vertex_velocity(mesh, index), scale)
        body_pos = wp.transform_point(X_bs, x_local)
        body_vel = wp.transform_vector(X_bs, v_local)
    else:
        edge_entry = rigid_edge_table[rigid_feature]
        index0 = edge_entry[1]
        index1 = edge_entry[2]
        sv0 = edge_indices[soft_feature, 2]
        sv1 = edge_indices[soft_feature, 3]
        corners = wp.vec3i(sv0, sv1, -1)
        r0_local = wp.cw_mul(wp.mesh_get_point(mesh, index0), scale)
        r1_local = wp.cw_mul(wp.mesh_get_point(mesh, index1), scale)
        r0_w = wp.transform_point(X_ws, r0_local)
        r1_w = wp.transform_point(X_ws, r1_local)
        s0 = particle_q[sv0]
        s1 = particle_q[sv1]
        std = wp.closest_point_edge_edge(r0_w, r1_w, s0, s1, edge_edge_parallel_epsilon)
        s = std[0]
        t = std[1]
        bary = wp.vec3(1.0 - t, t, 0.0)
        x_rigid = r0_w + s * (r1_w - r0_w)
        x_soft = s0 + t * (s1 - s0)
        e_rigid = r1_w - r0_w
        e_soft = s1 - s0
        cr = wp.cross(e_rigid, e_soft)
        cr_len = wp.length(cr)
        outward_w = transform_normal_with_scale(X_ws, scale, rigid_edge_outward_dirs[rigid_feature])
        if cr_len > edge_edge_parallel_epsilon * wp.length(e_rigid) * wp.length(e_soft):
            n_ref = cr / cr_len
            if wp.dot(n_ref, outward_w) < 0.0:
                n_ref = -n_ref
        else:
            n_ref = outward_w
        diff = x_soft - x_rigid
        normal = _oriented_contact_normal(diff, wp.length(diff), n_ref)
        v0 = _face_vertex_velocity(mesh, index0)
        v1 = _face_vertex_velocity(mesh, index1)
        body_pos = wp.transform_point(X_bs, r0_local + s * (r1_local - r0_local))
        body_vel = wp.transform_vector(X_bs, wp.cw_mul((1.0 - s) * v0 + s * v1, scale))

    soft_point = bary[0] * particle_q[corners[0]]
    if corners[1] >= 0:
        soft_point += bary[1] * particle_q[corners[1]]
    if corners[2] >= 0:
        soft_point += bary[2] * particle_q[corners[2]]
    rigid_point = wp.transform_point(X_ws, wp.transform_point(wp.transform_inverse(X_bs), body_pos))
    diff_world = soft_point - rigid_point
    distance_world = wp.length(diff_world)
    if distance_world > CONTACT_NORMAL_DEGENERATE_EPS:
        sign = wp.where((feature[0] & 8) != 0, -1.0, 1.0)
        normal = sign * diff_world / distance_world

    soft_contact_particle[tid] = particle
    soft_contact_indices[tid] = corners
    soft_contact_barycentric[tid] = bary
    soft_contact_body_pos[tid] = body_pos
    soft_contact_body_vel[tid] = body_vel
    soft_contact_normal[tid] = normal


def launch_soft_mesh_contacts(*, model: Model, state: State, contacts: Contacts, gap: float, data):
    """Append mesh contacts directly to final slots and evaluate their geometry."""
    device = model.device
    vt_pairs = data.vertex_pairs
    rigid_vertex_table, rigid_vertex_normals, rigid_edge_table, rigid_edge_outward_dirs = data.rigid_features
    detector = data.detector
    max_particle_radius = model.particle_max_radius
    if detector is not None:
        detector.vertex_positions = state.particle_q
        detector.refit_triangles()
        if model.edge_count:
            detector.refit_edges()
    contact_count = contacts.soft_contact_count
    features = contacts._soft_contact_mesh_features
    contact_shapes = contacts.soft_contact_shape
    n_vt = int(vt_pairs.shape[0])
    n_tv = int(rigid_vertex_table.shape[0])
    n_ee = int(rigid_edge_table.shape[0])
    contact_max = int(features.shape[0])
    if n_vt == 0 and n_tv == 0 and n_ee == 0:
        return

    cone_args = data.adjacency

    shape_args = [
        state.body_q,
        model.shape_transform,
        model.shape_body,
        model.shape_flags,
        model.shape_scale,
        model.shape_source_ptr,
    ]
    parallel_epsilon = detector.edge_edge_parallel_epsilon if detector is not None else 1.0e-5

    if n_vt > 0:
        wp.launch(
            _detect_mesh_vertex_contacts,
            dim=n_vt * _MESH_QUERY_LANES,
            inputs=[
                vt_pairs,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_flags,
                model.shape_scale,
                model.shape_source_ptr,
                model.shape_margin,
                gap,
                contact_max,
                *cone_args,
            ],
            outputs=[contact_count, features, contact_shapes],
            device=device,
            record_tape=False,
        )

    if detector is not None and n_tv > 0:
        wp.launch(
            _detect_mesh_face_contacts,
            dim=n_tv * _MESH_QUERY_LANES,
            inputs=[
                rigid_vertex_table,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.tri_indices,
                *shape_args,
                model.shape_world,
                model.shape_margin,
                detector.bvh_tris.id,
                detector.bvh_tris_group_roots,
                model.world_count,
                gap,
                max_particle_radius,
                contact_max,
                *cone_args,
            ],
            outputs=[contact_count, features, contact_shapes],
            device=device,
            record_tape=False,
        )

    if detector is not None and n_ee > 0:
        wp.launch(
            _detect_mesh_edge_contacts,
            dim=n_ee * _MESH_QUERY_LANES,
            inputs=[
                rigid_edge_table,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.edge_indices,
                *shape_args,
                model.shape_world,
                model.shape_margin,
                detector.bvh_edges.id,
                detector.bvh_edges_group_roots,
                model.world_count,
                parallel_epsilon,
                gap,
                max_particle_radius,
                contact_max,
                *cone_args,
            ],
            outputs=[contact_count, features, contact_shapes],
            device=device,
            record_tape=False,
        )

    if contact_max == 0:
        return

    wp.launch(
        _evaluate_mesh_contacts,
        dim=contact_max,
        inputs=[
            contact_count,
            features,
            contact_shapes,
            contact_max,
            model.shape_type,
            state.particle_q,
            model.tri_indices,
            model.edge_indices,
            state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_scale,
            model.shape_source_ptr,
            rigid_vertex_table,
            rigid_vertex_normals,
            rigid_edge_table,
            rigid_edge_outward_dirs,
            parallel_epsilon,
        ],
        outputs=[
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ],
        device=device,
    )


class MeshContactData:
    """Fixed mesh topology and acceleration structures; no candidate-contact buffer."""

    def __init__(self, model: Model, shape_mask: np.ndarray, vertex_pairs: wp.array[wp.vec2i]):
        self.vertex_pairs = vertex_pairs
        if model.tri_count:
            self.rigid_features = _build_rigid_features(model, shape_mask)
            if model.edge_count == 0:
                self.rigid_features = (
                    *self.rigid_features[:2],
                    wp.empty(0, dtype=wp.vec3i, device=model.device),
                    wp.empty(0, dtype=wp.vec3, device=model.device),
                )
            self.detector = TriMeshCollisionDetector(model)
        else:
            self.rigid_features = (
                wp.empty(0, dtype=wp.vec2i, device=model.device),
                wp.empty(0, dtype=wp.vec3, device=model.device),
                wp.empty(0, dtype=wp.vec3i, device=model.device),
                wp.empty(0, dtype=wp.vec3, device=model.device),
            )
            self.detector = None
        vertices, vertex_normals, edges, edge_normals = self.rigid_features
        if max(len(vertex_pairs), len(vertices), len(edges)) > np.iinfo(np.int32).max // _MESH_QUERY_LANES:
            raise ValueError("Mesh contact query lanes exceed 32-bit indexing capacity.")
        self.adjacency = [*_build_feature_adjacency(model, vertices, edges), vertex_normals, edge_normals]
        # This is a capacity estimate, not a bound; every write checks the final capacity.
        self.contact_capacity_hint = 4 * (len(vertex_pairs) + len(vertices) + len(edges))
