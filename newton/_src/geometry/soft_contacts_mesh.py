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

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .collision_core import transform_normal_with_scale
from .flags import MeshSignMethod, ParticleFlags, ShapeFlags
from .kernels import mesh_query_point_sign, resolve_mesh_sign_method, triangle_closest_point
from .soft_contacts_sdf import _shape_frames
from .tri_mesh_collision import TriMeshCollisionDetector
from .types import GeoType

if TYPE_CHECKING:
    from ..sim.contacts import Contacts
    from ..sim.model import Model
    from ..sim.state import State
    from .types import Mesh


# One CUDA warp cooperates on each particle query. CPU uses depth zero.
_MESH_QUERY_PARTITION_DEPTH = wp.constant(5)
_MESH_QUERY_PARTITIONS = wp.constant(1 << _MESH_QUERY_PARTITION_DEPTH)


@wp.func_native("""
const wp::Mesh mesh = wp::mesh_get(mesh_id);
int root = *mesh.bvh.root;
if (root < 0) return -2;
for (int level = 0; level < depth; ++level) {
    const auto lower = wp::bvh_load_node(mesh.bvh.node_lowers, root);
    if (lower.b) return (lane & ((1 << (depth - level)) - 1)) == 0 ? root : -2;
    const auto upper = wp::bvh_load_node(mesh.bvh.node_uppers, root);
    root = ((lane >> (depth - 1 - level)) & 1) ? upper.i : lower.i;
}
return root;
""")
def _mesh_query_partition(mesh_id: wp.uint64, lane: int, depth: int) -> int:
    """Partition the existing mesh BVH; -2 marks an unused shallow-tree lane."""
    ...


@wp.func_native("""
int sign = 0;
#if defined(__CUDA_ARCH__)
const unsigned mask = __activemask();
const int leader = (threadIdx.x & 31) & ~(lanes - 1);
if ((threadIdx.x & (lanes - 1)) == 0) {
#endif
    const auto query = parity ? wp::mesh_query_point_sign_parity(mesh_id, point, radius)
                              : wp::mesh_query_point_sign_normal(mesh_id, point, radius);
    sign = query.result ? (query.sign < 0.0f ? -1 : 1) : 0;
#if defined(__CUDA_ARCH__)
}
return __shfl_sync(mask, sign, leader);
#else
return sign;
#endif
""")
def _mesh_partition_sign(mesh_id: wp.uint64, point: wp.vec3, radius: float, lanes: int, parity: bool) -> int:
    """Share an exact nearest query within a power-of-two group of CUDA lanes.

    ``parity`` selects ray-crossing parity (watertight meshes) instead of the pseudo-normal sign.

    All lanes in each group must participate with identical query arguments.
    Group size must divide the CUDA warp size and the launch block dimension.
    """
    ...


@wp.func
def _feature_query_capsule(
    scale: wp.vec3,
    transform: wp.transform,
    bounds: wp.vec4,
    error: wp.vec3,
    radius: float,
):
    s = wp.max(wp.abs(scale[0]), wp.max(wp.abs(scale[1]), wp.abs(scale[2])))
    minimum = wp.min(wp.abs(scale[0]), wp.min(wp.abs(scale[1]), wp.abs(scale[2])))
    padding = error[0] * s * s + error[1] * s * radius + error[2] * radius * radius
    axis = wp.normalize(wp.transform_vector(transform, wp.cw_div(wp.vec3(bounds[0], bounds[1], bounds[2]), scale)))
    width = (bounds[3] * s * radius + padding) / minimum
    return axis, wp.min(width, radius)


@lru_cache(maxsize=16)
def _cone_pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Reuse pair indices for at most six cone directions or sixteen axes."""
    return np.triu_indices(count, 1)


def _cone_query_bounds(
    point: np.ndarray, positions: np.ndarray, *, edge: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Bound both validity cones by an axis and relative capsule radius.

    Opposing halfspaces imply a slab. Two independent slabs bound transverse
    distance by their smallest Gram eigenvalue. Dropping constraints only
    widens this bound. The returned polynomial propagates the same rounding
    allowance as ``_cone_valid`` through those slab inequalities.
    """
    directions = positions - point
    length = np.linalg.norm(directions, axis=1)
    point_bound = np.linalg.norm(point)
    fallback = np.array([0.0, 0.0, 1.0, 1.0])
    if edge is not None:
        half_length = np.linalg.norm(edge) * 0.5
        if half_length == 0:
            return fallback, np.zeros(3)
        axis = edge / (2 * half_length)
        directions -= (directions @ axis)[:, None] * axis
        length += half_length
        point_bound += half_length
    projected = np.linalg.norm(directions, axis=1)
    valid = projected > 0
    directions = directions[valid] / projected[valid, None]
    p = point_bound + np.linalg.norm(positions[valid], axis=1)
    errors = 2.0e-6 * np.column_stack((p * length[valid], p + length[valid], np.ones(len(p)))) / projected[valid, None]
    if len(directions) < 2:
        return fallback, np.zeros(3)
    # Six extremal halfspaces suffice for a conservative accelerator. Limit
    # precomputation storage/work even at arbitrarily high-valence vertices;
    # the exact acceptance test still uses every incident neighbor.
    if len(directions) > 6:
        support = np.unique(np.concatenate((np.argmin(directions, axis=0), np.argmax(directions, axis=0))))
        directions, errors = directions[support], errors[support]
    i, j = _cone_pair_indices(len(directions))
    delta = directions[i] - directions[j]
    a = np.linalg.norm(delta, axis=1)
    b = np.linalg.norm(directions[i] + directions[j], axis=1)
    valid = a > b
    if not np.any(valid):
        return fallback, np.zeros(3)
    i, j, delta, a, b = i[valid], j[valid], delta[valid], a[valid], b[valid]
    axes = delta / a[:, None]
    width = b / a
    relaxation = 2 * np.maximum(errors[i], errors[j]) / a[:, None]
    if edge is not None:
        axes = np.vstack((axes, axis))
        width = np.append(width, 0.0)
        relaxation = np.vstack((relaxation, np.zeros(3)))
    i, j = _cone_pair_indices(len(axes))
    eigenvalue = 1 - np.abs(np.sum(axes[i] * axes[j], axis=1))
    valid = eigenvalue > 1.0e-12
    if not np.any(valid):
        return fallback, np.zeros(3)
    i, j, eigenvalue = i[valid], j[valid], eigenvalue[valid]
    widths = np.sqrt((width[i] ** 2 + width[j] ** 2) / eigenvalue)
    best = np.argmin(widths)
    if widths[best] >= 1.0:
        return fallback, np.zeros(3)
    direction = np.cross(axes[i[best]], axes[j[best]])
    direction /= np.linalg.norm(direction)
    error = (relaxation[i[best]] + relaxation[j[best]]) / np.sqrt(eigenvalue[best])
    return np.append(direction, widths[best] + 1.0e-5), error


def _cone_query_bounds_batch(
    points: np.ndarray, positions: np.ndarray, *, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``_cone_query_bounds`` for equally sized neighbor lists.

    Inputs have shapes (N, 3), (N, K, 3), and optionally (N, 3).
    Masked candidates retain the scalar implementation's order, including its
    first-minimum tie break. The caller batches by valence and limits N to keep
    temporary storage bounded. All computation remains float64.
    """
    count, neighbor_count = positions.shape[:2]
    bounds = np.tile([0.0, 0.0, 1.0, 1.0], (count, 1))
    result_errors = np.zeros((count, 3))
    if count == 0 or neighbor_count < 2:
        return bounds, result_errors

    directions = positions - points[:, None, :]
    length = np.linalg.norm(directions, axis=2)
    # Scalar np.linalg.norm(vec3) uses a dot product, whereas norm(..., axis=1)
    # sums three squares. Keep the dot-product evaluation order when batching
    # those particular norms so that rounding agrees with the scalar version.
    point_bound = np.sqrt((points[:, None, :] @ points[:, :, None])[:, 0, 0])
    if edges is not None:
        half_length = np.sqrt((edges[:, None, :] @ edges[:, :, None])[:, 0, 0]) * 0.5
        axis = edges / np.where(half_length > 0, 2 * half_length, 1.0)[:, None]
        directions -= (directions @ axis[:, :, None]) * axis[:, None, :]
        length += half_length[:, None]
        point_bound += half_length
    projected = np.linalg.norm(directions, axis=2)
    valid = projected > 0
    if edges is not None:
        valid &= half_length[:, None] > 0
    denominator = np.where(valid, projected, 1.0)
    directions /= denominator[:, :, None]
    p = point_bound[:, None] + np.linalg.norm(positions, axis=2)
    errors = 2.0e-6 * np.stack((p * length, p + length, np.ones_like(p)), axis=2) / denominator[:, :, None]

    if neighbor_count > 6:
        # Select the same six coordinate extrema as the scalar implementation,
        # keeping unique indices in their original order. Rows with at most six
        # nonzero directions retain all of them.
        extrema = np.concatenate(
            (
                np.argmin(np.where(valid[:, :, None], directions, np.inf), axis=1),
                np.argmax(np.where(valid[:, :, None], directions, -np.inf), axis=1),
            ),
            axis=1,
        )
        selected = np.zeros_like(valid)
        np.put_along_axis(selected, extrema, True, axis=1)
        selected = valid & np.where((np.sum(valid, axis=1) > 6)[:, None], selected, True)
        indices = np.sort(np.where(selected, np.arange(neighbor_count), neighbor_count), axis=1)[:, :6]
        valid = indices < neighbor_count
        indices = np.minimum(indices, neighbor_count - 1)
        directions = np.take_along_axis(directions, indices[:, :, None], axis=1)
        errors = np.take_along_axis(errors, indices[:, :, None], axis=1)

    i, j = _cone_pair_indices(directions.shape[1])
    delta = directions[:, i] - directions[:, j]
    a = np.linalg.norm(delta, axis=2)
    b = np.linalg.norm(directions[:, i] + directions[:, j], axis=2)
    active = valid[:, i] & valid[:, j] & (a > b)
    denominator = np.where(active, a, 1.0)
    axes = delta / denominator[:, :, None]
    width = b / denominator
    relaxation = 2 * np.maximum(errors[:, i], errors[:, j]) / denominator[:, :, None]
    if edges is not None:
        axes = np.concatenate((axes, axis[:, None, :]), axis=1)
        width = np.column_stack((width, np.zeros(count)))
        relaxation = np.concatenate((relaxation, np.zeros((count, 1, 3))), axis=1)
        active = np.column_stack((active, half_length > 0))

    i, j = _cone_pair_indices(axes.shape[1])
    if len(i) == 0:
        return bounds, result_errors
    eigenvalue = 1 - np.abs(np.sum(axes[:, i] * axes[:, j], axis=2))
    valid = active[:, i] & active[:, j] & (eigenvalue > 1.0e-12)
    widths = np.sqrt((width[:, i] ** 2 + width[:, j] ** 2) / np.where(valid, eigenvalue, 1.0))
    widths = np.where(valid, widths, np.inf)
    best = np.argmin(widths, axis=1)
    rows = np.flatnonzero(widths[np.arange(count), best] < 1.0)
    best = best[rows]
    direction = np.cross(axes[rows, i[best]], axes[rows, j[best]])
    direction /= np.sqrt((direction[:, None, :] @ direction[:, :, None])[:, 0, 0])[:, None]
    bounds[rows, :3] = direction
    bounds[rows, 3] = widths[rows, best] + 1.0e-5
    result_errors[rows] = (relaxation[rows, i[best]] + relaxation[rows, j[best]]) / np.sqrt(
        eigenvalue[rows, best, None]
    )
    return bounds, result_errors


@wp.func
def _segment_triangle_parameter(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
    """Parameter along segment ``pq`` where it passes through triangle ``abc``, or -1."""
    direction = q - p
    ab = b - a
    ac = c - a
    h = wp.cross(direction, ac)
    det = wp.dot(ab, h)
    if det == 0.0:
        return float(-1.0)
    offset = p - a
    u = wp.dot(offset, h) / det
    k = wp.cross(offset, ab)
    v = wp.dot(direction, k) / det
    t = wp.dot(ac, k) / det
    if u < 0.0 or v < 0.0 or u + v > 1.0 or t < 0.0 or t > 1.0:
        return float(-1.0)
    return t


@wp.func
def _is_inside(mesh: wp.uint64, X_sw: wp.transform, scale: wp.vec3, point: wp.vec3, sign_method: int) -> bool:
    """Whether a world point lies inside the mesh."""
    query = mesh_query_point_sign(mesh, wp.cw_div(wp.transform_point(X_sw, point), scale), 1.0e6, sign_method)
    return query.result and query.sign < 0.0


@wp.func
def _chord_fraction(mesh: wp.uint64, X_sw: wp.transform, scale: wp.vec3, origin: wp.vec3, target: wp.vec3, face: int):
    """Fraction from ``origin`` toward ``target`` of the first surface hit other than ``face``.

    Returns 0 when the first hit along the segment is ``face`` itself, or nothing is hit, so the
    stretch between ``origin`` and ``face`` crosses no other surface.
    """
    start = wp.cw_div(wp.transform_point(X_sw, origin), scale)
    delta = wp.cw_div(wp.transform_point(X_sw, target), scale) - start
    length = wp.length(delta)
    if length == 0.0:
        return float(0.0)
    query = wp.mesh_query_ray(mesh, start, delta / length, length)
    if not query.result or query.face == face:
        return float(0.0)
    return query.t / length


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


def _collision_meshes(model: Model, shape_mask: np.ndarray) -> dict[int, Mesh]:
    """Return the mesh backing each masked shape's ``shape_source_ptr`` and edge table.

    Convex hulls collide against a vertex-deduplicated copy of their source, so its packed
    collision edges use that copy's vertex numbering, not ``model.shape_source``'s.
    """
    from ..sim.builder import _deduplicate_convex_collision_mesh  # noqa: PLC0415

    shape_types = model.shape_type.numpy()
    convex_copies: dict[int, Mesh] = {}
    meshes = {}
    for shape in np.flatnonzero(shape_mask):
        mesh = model.shape_source[shape]
        if mesh is None or not hasattr(mesh, "indices"):
            raise ValueError(f"mesh/convex shape {int(shape)} has no shape_source Mesh")
        if shape_types[shape] == int(GeoType.CONVEX_MESH):
            if id(mesh) not in convex_copies:
                convex_copies[id(mesh)] = _deduplicate_convex_collision_mesh(mesh)
            mesh = convex_copies[id(mesh)]
        meshes[int(shape)] = mesh
    return meshes


def _geometry_key(mesh: Mesh) -> tuple[int, int, int]:
    """Share per-mesh precomputation across distinct but identical Mesh objects."""
    return (len(mesh.vertices), len(mesh.indices), hash(mesh))


def _edge_rows(keys: np.ndarray, query: np.ndarray, what: str) -> np.ndarray:
    """Locate sorted edge keys, rejecting edges that are not part of the triangle topology."""
    rows = np.searchsorted(keys, query)
    if np.any(rows >= len(keys)) or np.any(keys[np.minimum(rows, len(keys) - 1)] != query):
        raise ValueError(f"{what} contains an edge that is not an edge of the mesh triangles.")
    return rows


def _max_concurrent_components(lower: np.ndarray, upper: np.ndarray, band: float) -> int:
    """Bound how many disjoint surface components a single point can touch within ``band``.

    A point within ``band`` of several components lies in all of their band-expanded bounds,
    so those bounds overlap pairwise. The most bounds overlapping any one bound, counting
    itself, therefore bounds the components that one particle can contact at once.
    """
    lower = lower - band
    upper = upper + band
    most = 0
    for start in range(0, len(lower), 1024):
        rows = slice(start, start + 1024)
        overlap = np.all((lower[rows, None] <= upper[None]) & (lower[None] <= upper[rows, None]), axis=2)
        most = max(most, int(overlap.sum(axis=1).max()))
    return most


def _build_feature_adjacency(
    model: Model, meshes: dict[int, Mesh], vertex_table: wp.array, edge_table: wp.array, contact_band: float
):
    """Build fixed incident-feature spans and ownership.

    Also returns, per shape, how many disjoint surface components one particle can touch within
    ``contact_band`` plus the shape's margin.
    """
    et = edge_table.numpy()
    offsets = np.zeros(model.shape_count, dtype=np.int32)
    concurrent_components = np.zeros(model.shape_count, dtype=np.int32)
    shape_scale = model.shape_scale.numpy()
    shape_margin = model.shape_margin.numpy() if model.shape_margin is not None else np.zeros(model.shape_count)
    concurrency_cache = {}
    vertex_spans, edge_spans, neighbors = [], [], []
    vertex_bounds, vertex_errors, edge_bounds, edge_errors = [], [], [], []
    ee = np.zeros(len(et), dtype=np.int32)
    cache = {}

    def build(mesh):
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

        unvisited = set(incident)
        component_of = {}
        component_count = 0
        while unvisited:
            pending = [unvisited.pop()]
            component_of[pending[0]] = component_count
            while pending:
                connected = incident[pending.pop()] & unvisited
                unvisited.difference_update(connected)
                component_of.update(dict.fromkeys(connected, component_count))
                pending.extend(connected)
            component_count += 1
        vertex_positions = np.asarray(mesh.vertices, dtype=np.float64)[idx]
        component_ids = np.fromiter((component_of[int(v)] for v in canon), dtype=np.int64, count=len(canon))
        component_lower = np.full((component_count, 3), np.inf)
        component_upper = np.full((component_count, 3), -np.inf)
        np.minimum.at(component_lower, component_ids, vertex_positions)
        np.maximum.at(component_upper, component_ids, vertex_positions)

        def span(vertices, owner, representatives=representative):
            start = len(neighbors)
            neighbors.extend(representatives[v] for v in sorted(vertices))
            return (start, len(neighbors), owner)

        vs = {v: span(n, representative[v] // 3) for v, n in incident.items()}
        es = {key: span(n, edge_owner[key]) for key, n in opposite.items()}
        points = np.asarray(mesh.vertices, dtype=np.float64)[idx]

        def feature_bounds(features, *, is_edge, enabled):
            if not enabled:
                return dict.fromkeys(features, (np.array([0.0, 0.0, 1.0, 1.0]), np.zeros(3)))
            groups = {}
            for key, ns in features.items():
                groups.setdefault(len(ns), []).append(key)
            result = {}
            for valence, keys in groups.items():
                # At most six directions enter the axis-pair search. Also cap
                # neighbor storage for unusually high-valence/nonmanifold meshes.
                batch_size = max(1, min(2048, 65536 // max(valence, 1)))
                for start in range(0, len(keys), batch_size):
                    batch = keys[start : start + batch_size]
                    ns = np.asarray([[representative[n] for n in features[key]] for key in batch], dtype=np.int32)
                    if is_edge:
                        endpoints = points[[[representative[v] for v in key] for key in batch]]
                        centers = (endpoints[:, 0] + endpoints[:, 1]) * 0.5
                        edges = endpoints[:, 1] - endpoints[:, 0]
                    else:
                        centers = points[[representative[key] for key in batch]]
                        edges = None
                    bounds, errors = _cone_query_bounds_batch(centers, points[ns], edges=edges)
                    result.update((key, (bounds[row], errors[row])) for row, key in enumerate(batch))
            return result

        vb = feature_bounds(incident, is_edge=False, enabled=len(vertex_table) > 0)
        eb = feature_bounds(opposite, is_edge=True, enabled=len(et) > 0)
        edge_slots = {}
        offset = len(vertex_spans)
        for slot, v in enumerate(canon):
            vertex_spans.append(vs[int(v)])
            vertex_bounds.append(vb[int(v)][0])
            vertex_errors.append(vb[int(v)][1])
            base, k = (slot // 3) * 3, slot % 3
            key = tuple(sorted((int(canon[base + (k + 1) % 3]), int(canon[base + (k + 2) % 3]))))
            edge_spans.append(es[key])
            edge_bounds.append(eb[key][0])
            edge_errors.append(eb[key][1])
            edge_slots.setdefault(key, offset + slot)
        keys = sorted(es)
        edge_keys = np.asarray([(a << 32) | b for a, b in keys], dtype=np.int64)
        edge_data = np.asarray([edge_slots[key] for key in keys], dtype=np.int32)
        return offset, canon, edge_keys, edge_data, (component_lower, component_upper)

    # Shape instances share immutable local topology. Only the feature rows
    # carry a shape id; constructing full adjacency per world is unnecessary.
    for shape, mesh in meshes.items():
        key = _geometry_key(mesh)
        if key not in cache:
            cache[key] = build(mesh)
        offset, canon, edge_keys, edge_data, (component_lower, component_upper) = cache[key]
        offsets[shape] = offset
        scale = shape_scale[shape].astype(np.float64)
        band = contact_band + float(shape_margin[shape])
        concurrency_key = (key, tuple(scale), band)
        if concurrency_key not in concurrency_cache:
            scaled_lower = np.minimum(component_lower * scale, component_upper * scale)
            scaled_upper = np.maximum(component_lower * scale, component_upper * scale)
            concurrency_cache[concurrency_key] = _max_concurrent_components(scaled_lower, scaled_upper, band)
        concurrent_components[shape] = concurrency_cache[concurrency_key]
        start, end = np.searchsorted(et[:, 0], (shape, shape + 1))
        edge_canon = np.sort(canon[et[start:end, 1:]].astype(np.int64), axis=1)
        keys = (edge_canon[:, 0] << 32) | edge_canon[:, 1]
        ee[start:end] = edge_data[_edge_rows(edge_keys, keys, "Rigid edge table")]
    if max(len(vertex_spans), len(edge_spans), len(neighbors)) > np.iinfo(np.int32).max:
        raise ValueError("Mesh contact topology exceeds 32-bit indexing capacity.")
    arrays = [wp.array(offsets, dtype=int, device=model.device)]
    arrays.extend(
        wp.array(np.asarray(x, dtype=np.int32).reshape(-1, 3), dtype=wp.vec3i, device=model.device)
        for x in (vertex_spans, edge_spans)
    )
    arrays.append(wp.array(ee, dtype=int, device=model.device))
    arrays.append(wp.array(neighbors, dtype=int, device=model.device))
    arrays.extend(
        wp.array(np.asarray(x, dtype=np.float32).reshape(-1, width), dtype=dtype, device=model.device)
        for x, width, dtype in (
            (vertex_bounds, 4, wp.vec4),
            (vertex_errors, 3, wp.vec3),
            (edge_bounds, 4, wp.vec4),
            (edge_errors, 3, wp.vec3),
        )
    )
    return arrays, concurrent_components


CONTACT_NORMAL_DEGENERATE_EPS = wp.constant(1.0e-6)


def _mesh_feature_data(
    mesh: Mesh, *, collision_edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique face-vertex slots, edges, and outward reference normals."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    idx = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
    if idx.size == 0 or verts.size == 0:
        empty_i = np.empty(0, dtype=np.int32)
        return empty_i, np.empty((0, 3), np.float32), np.empty((0, 2), np.int32), np.empty((0, 3), np.float32)
    tris = idx.reshape(-1, 3)
    canonical = mesh._canonical_vertex_ids()
    n_canon = int(canonical.max()) + 1

    orig_edges, slot_keys, _sort_order, _keys_sorted, face_normals, face_norms = mesh._build_edge_slot_topology(
        canonical
    )
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
    edge_keys, first_idx, inverse = np.unique(slot_keys, return_index=True, return_inverse=True)
    edge_normal_accumulation = np.zeros((len(first_idx), 3), dtype=np.float64)
    np.add.at(edge_normal_accumulation, inverse, np.repeat(fn_unit, 3, axis=0))
    edge_outward = _unit(edge_normal_accumulation).astype(np.float32)
    if collision_edges is None:
        edge_canon = canonical[orig_edges[first_idx]]
    else:
        # Keep exactly the SDF edge set, including an intentionally empty set.
        # Full triangle adjacency still supplies the normals and validity cones.
        edge_canon = canonical[collision_edges]
        sorted_canon = np.sort(edge_canon.astype(np.int64), axis=1)
        selected_keys = (sorted_canon[:, 0] << 32) | sorted_canon[:, 1]
        edge_outward = edge_outward[_edge_rows(edge_keys, selected_keys, "Mesh collision edge set")]
    edge_table = np.column_stack((index_of_canon[edge_canon[:, 0]], index_of_canon[edge_canon[:, 1]])).astype(np.int32)

    return vertex_table, vertex_normals, edge_table, edge_outward


def _build_rigid_features(
    model: Model, meshes: dict[int, Mesh]
) -> tuple[wp.array[wp.vec2i], wp.array[wp.vec3], wp.array[wp.vec3i], wp.array[wp.vec3]]:

    device = model.device
    vertex_rows: list[np.ndarray] = []
    vertex_normals: list[np.ndarray] = []
    edge_rows: list[np.ndarray] = []
    edge_outwards: list[np.ndarray] = []
    cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    edge_ranges = model.shape_edge_range.numpy()
    packed_edges = model.mesh_edge_indices.numpy()

    for shape, mesh in meshes.items():
        start, count = (int(value) for value in edge_ranges[shape])
        # Finalized tables include SDF edges cooked by the builder, which need
        # not be attached to the source Mesh. Slices may differ for one source.
        collision_edges = packed_edges[start : start + count] if start >= 0 else mesh._collision_edges
        key = (_geometry_key(mesh), None if collision_edges is None else np.asarray(collision_edges).tobytes())
        data = cache.get(key)
        if data is None:
            data = _mesh_feature_data(mesh, collision_edges=collision_edges)
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
# Edge penetration recovery: the middle of a soft-edge chord inside the solid, whose soft-edge
# parameter is stored with the record, pairs with its nearest surface point. Bit 3 marks penetration.
_MESH_FEATURE_EE_DEPTH = wp.constant(3)


@wp.func
def _append_mesh_contact(
    family: wp.int32,
    soft_feature: wp.int32,
    rigid_shape: wp.int32,
    rigid_feature: wp.int32,
    contact_max: wp.int32,
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
) -> int:
    """Append a feature record, returning its final slot or -1 when capacity is exhausted."""
    index = wp.atomic_add(contact_count, 0, wp.int64(1))
    if index == wp.int64(contact_max):
        wp.printf("Mesh soft-contact capacity exceeded (%d); this collision result is incomplete.\n", contact_max)
    if index < wp.int64(contact_max):
        features[index] = wp.vec3i(family, soft_feature, rigid_feature)
        contact_shapes[index] = rigid_shape
        return wp.int32(index)
    return -1


@wp.kernel(enable_backward=False)
def _begin_mesh_contacts(source: wp.array[int], destination: wp.array[wp.int64]):
    destination[0] = wp.int64(source[0])


@wp.kernel(enable_backward=False)
def _end_mesh_contacts(source: wp.array[wp.int64], destination: wp.array[int], capacity: int):
    destination[0] = wp.int32(wp.min(source[0], wp.int64(capacity) + wp.int64(1)))


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
def _feature_mesh_sign(
    mesh: wp.uint64, X_sw: wp.transform, scale: wp.vec3, soft: wp.vec3, sign_method: int, feature_distance: float
):
    """Sign a soft feature point: -1 inside, 1 outside, 2 on the surface, 0 without a surface.

    A soft edge crossing a rigid face has its closest point to that face's boundary edges exactly
    on the face, where a global inside/outside query only reports roundoff; the caller then
    resolves the side from local feature geometry. The on-surface tolerance, in world units,
    covers rounding of the world-to-mesh transform and separations negligible against the pair's
    feature distance.
    """
    query_point = wp.cw_div(wp.transform_point(X_sw, soft), scale)
    query = mesh_query_point_sign(mesh, query_point, 1.0e6, sign_method)
    if not query.result:
        return int(0)
    surface = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    separation = wp.length(wp.cw_mul(query_point - surface, scale))
    if separation <= 2.0e-6 * (wp.length(soft) + wp.length(X_sw.p)) + 1.0e-4 * feature_distance:
        return int(2)
    return wp.where(query.sign < 0.0, -1, 1)


@wp.func
def _local_side(outward: bool, inward: bool, separation: wp.vec3, reference: wp.vec3) -> int:
    """Side of an on-surface soft point: an exclusive validity cone decides, else ``reference``."""
    if outward != inward:
        return wp.where(inward, -1, 1)
    return wp.where(wp.dot(separation, reference) < 0.0, -1, 1)


@wp.kernel(enable_backward=False)
def _detect_mesh_vertex_contacts(
    vt_pairs: wp.array[wp.vec2i],
    partition_depth: int,
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32],
    shape_lower: wp.array[wp.vec3],
    shape_upper: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    gap: float,
    contact_max: wp.int32,
    face_offsets: wp.array[int],
    vertex_spans: wp.array[wp.vec3i],
    edge_spans: wp.array[wp.vec3i],
    rigid_edge_slots: wp.array[int],
    neighbors: wp.array[int],
    vertex_bounds: wp.array[wp.vec4],
    vertex_errors: wp.array[wp.vec3],
    edge_bounds: wp.array[wp.vec4],
    edge_errors: wp.array[wp.vec3],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    partitions = 1 << partition_depth
    tid = wp.tid() >> partition_depth
    lane = wp.tid() & (partitions - 1)
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
    lower_bound = shape_lower[shape_index] - wp.vec3(threshold)
    upper_bound = shape_upper[shape_index] + wp.vec3(threshold)
    if (
        x_local[0] < lower_bound[0]
        or x_local[1] < lower_bound[1]
        or x_local[2] < lower_bound[2]
        or x_local[0] > upper_bound[0]
        or x_local[1] > upper_bound[1]
        or x_local[2] > upper_bound[2]
    ):
        return

    mesh = shape_source_ptr[shape_index]
    min_scale = wp.min(wp.min(wp.abs(scale[0]), wp.abs(scale[1])), wp.abs(scale[2]))
    x_mesh = wp.cw_div(x_local, scale)
    r_mesh = threshold / min_scale
    sign_method = resolve_mesh_sign_method(shape_mesh_properties[shape_index])
    vertex_sign = _mesh_partition_sign(mesh, x_mesh, r_mesh, partitions, sign_method == MeshSignMethod.PARITY)
    if vertex_sign == 0:
        if lane != 0:
            return
        # A finite proximity band must not discard already penetrating points.
        # The expanded shape bounds have already rejected distant particles.
        recovery_radius = wp.length(upper_bound - lower_bound) / min_scale
        recovery = mesh_query_point_sign(mesh, x_mesh, recovery_radius, sign_method)
        if recovery.result and recovery.sign < 0.0:
            _append_mesh_contact(
                _MESH_FEATURE_VT + 8,
                particle_index,
                shape_index,
                recovery.face,
                contact_max,
                contact_count,
                features,
                contact_shapes,
            )
        return
    root = _mesh_query_partition(mesh, lane, partition_depth)
    if root == -2:
        return
    query = wp.bvh_query_sphere(wp.mesh_get_bvh(mesh), x_mesh, r_mesh, root)
    face = wp.int32(0)
    while wp.bvh_query_next(query, face):
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
            _append_mesh_contact(
                _MESH_FEATURE_VT + wp.where(vertex_sign < 0, 8, 0),
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
    shape_mesh_properties: wp.array[wp.int32],
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
    rigid_edge_slots: wp.array[int],
    neighbors: wp.array[int],
    vertex_bounds: wp.array[wp.vec4],
    vertex_errors: wp.array[wp.vec3],
    edge_bounds: wp.array[wp.vec4],
    edge_errors: wp.array[wp.vec3],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    tid = wp.tid()
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
    slot = face_offsets[shape_index] + index
    axis, width = _feature_query_capsule(scale, X_ws, vertex_bounds[slot], vertex_errors[slot], bound)
    half_length = wp.where(width < bound, bound, 0.0)
    start = x_w - half_length * axis

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
                query = wp.bvh_query_capsule(bvh_tris_id, start, axis, width)
            else:
                query = wp.bvh_query_capsule(bvh_tris_id, start, axis, width, group_root)

            tri_index = wp.int32(0)
            while wp.bvh_query_next(query, tri_index, 2.0 * half_length):
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
                    outward = _cone_valid(mesh, scale, x_local, diff, vertex_spans[slot], neighbors)
                    inward = _cone_valid(mesh, scale, x_local, -diff, vertex_spans[slot], neighbors)
                    if not outward and not inward:
                        continue
                    sign = _feature_mesh_sign(
                        mesh,
                        _X_sw,
                        scale,
                        cp,
                        resolve_mesh_sign_method(shape_mesh_properties[shape_index]),
                        wp.length(cp - x_w),
                    )
                    if sign == 2:
                        reference = transform_normal_with_scale(X_ws, scale, vertex_outward[tid])
                        sign = _local_side(outward, inward, cp - x_w, reference)
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
    shape_mesh_properties: wp.array[wp.int32],
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
    rigid_edge_slots: wp.array[int],
    neighbors: wp.array[int],
    vertex_bounds: wp.array[wp.vec4],
    vertex_errors: wp.array[wp.vec3],
    edge_bounds: wp.array[wp.vec4],
    edge_errors: wp.array[wp.vec3],
    vertex_outward: wp.array[wp.vec3],
    edge_outward: wp.array[wp.vec3],
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    contact_shapes: wp.array[int],
):

    tid = wp.tid()
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
    slot = rigid_edge_slots[tid]
    axis, width = _feature_query_capsule(scale, X_ws, edge_bounds[slot], edge_errors[slot], bound)
    half_length = bound + 0.5 * wp.length(r1_w - r0_w)
    width += 0.5 * wp.length(r1_w - r0_w)
    start = 0.5 * (r0_w + r1_w) - half_length * axis
    lower -= wp.vec3(bound)
    upper += wp.vec3(bound)
    capsule_size = 2.0 * (wp.vec3(wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2])) * half_length + wp.vec3(width))
    box_size = upper - lower
    use_capsule = capsule_size[0] * capsule_size[1] * capsule_size[2] < box_size[0] * box_size[1] * box_size[2]

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
                group_root = -1
            if use_capsule:
                query = wp.bvh_query_capsule(bvh_edges_id, start, axis, width, group_root)
            else:
                query = wp.bvh_query_aabb(bvh_edges_id, lower, upper, group_root)

            edge_index = wp.int32(0)
            while wp.bvh_query_next(query, edge_index, 2.0 * half_length):
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
                    outward = _cone_valid(mesh, scale, rigid_local, diff_local, edge_spans[slot], neighbors)
                    inward = _cone_valid(mesh, scale, rigid_local, -diff_local, edge_spans[slot], neighbors)
                    if not outward and not inward:
                        continue
                    sign = _feature_mesh_sign(
                        mesh,
                        _X_sw,
                        scale,
                        soft_point,
                        resolve_mesh_sign_method(shape_mesh_properties[shape_index]),
                        std[2],
                    )
                    if sign == 2:
                        reference = transform_normal_with_scale(X_ws, scale, edge_outward[tid])
                        sign = _local_side(outward, inward, soft_point - rigid_point, reference)
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


@wp.kernel(enable_backward=False)
def _detect_mesh_edge_penetrations(
    rigid_face_table: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    edge_indices: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    bvh_edges_id: wp.uint64,
    bvh_edges_group_roots: wp.array[wp.int32],
    world_count: wp.int32,
    contact_max: wp.int32,
    contact_count: wp.array[wp.int64],
    features: wp.array[wp.vec3i],
    params: wp.array[float],
    contact_shapes: wp.array[int],
):
    """Recover soft edges that pass through a mesh, including deeper than feature pairs track.

    A gripper can pinch a soft edge, whose vertices stay outside, deeper into a pad than the band
    that feature pairs search. Such an edge enters and leaves the solid through rigid triangles;
    the contact pairs the middle of that chord with its nearest surface point (see evaluation).
    Each triangle searches only its own bounds, so scenes without crossings stay cheap.
    """
    tid = wp.tid()
    entry = rigid_face_table[tid]
    shape_index = entry[0]
    face = entry[1]
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    scale = shape_scale[shape_index]
    _X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape_index)
    mesh = shape_source_ptr[shape_index]
    a = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 0), scale))
    b = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 1), scale))
    c = wp.transform_point(X_ws, wp.cw_mul(wp.mesh_get_point(mesh, face * 3 + 2), scale))
    method = resolve_mesh_sign_method(shape_mesh_properties[shape_index])
    lower = wp.min(a, wp.min(b, c))
    upper = wp.max(a, wp.max(b, c))

    rigid_world = shape_world[shape_index]
    for query_pass in range(2):
        group_root = wp.int32(-1)
        run_query = rigid_world < 0 and query_pass == 0
        if rigid_world >= 0:
            group_root = bvh_edges_group_roots[wp.where(query_pass == 0, rigid_world, world_count)]
            run_query = group_root >= 0
        if run_query:
            query = wp.bvh_query_aabb(bvh_edges_id, lower, upper, group_root)
            edge_index = wp.int32(0)
            while wp.bvh_query_next(query, edge_index):
                sv0 = edge_indices[edge_index, 2]
                sv1 = edge_indices[edge_index, 3]
                active = (particle_flags[sv0] & ParticleFlags.ACTIVE) | (particle_flags[sv1] & ParticleFlags.ACTIVE)
                if active == 0:
                    continue
                crossing = _segment_triangle_parameter(particle_q[sv0], particle_q[sv1], a, b, c)
                if crossing < 0.0:
                    continue
                # The soft edge lies inside the solid along chords between crossings. A chord that
                # reaches an endpoint means that particle is inside, which the vertex pass recovers;
                # a chord bounded by two crossings is emitted once, by its lower crossing. Its
                # midpoint is the deepest point of a slab such as a pinching pad.
                soft0 = particle_q[sv0]
                soft1 = particle_q[sv1]
                remaining = _chord_fraction(mesh, X_sw, scale, soft1, soft0, face)
                if remaining == 0.0:
                    continue
                t = 0.5 * (crossing + 1.0 - remaining)
                if t > crossing and _is_inside(mesh, X_sw, scale, soft0 + t * (soft1 - soft0), method):
                    slot = _append_mesh_contact(
                        _MESH_FEATURE_EE_DEPTH + 8,
                        edge_index,
                        shape_index,
                        tid,
                        contact_max,
                        contact_count,
                        features,
                        contact_shapes,
                    )
                    if slot >= 0:
                        params[slot] = t


@wp.func
def _nearest_surface_contact(
    mesh: wp.uint64, X_bs: wp.transform, X_sw: wp.transform, scale: wp.vec3, point: wp.vec3, sign_method: int
):
    """Body-frame position and velocity of the mesh surface point nearest to a world point.

    Also returns that face's shape-frame normal, the fallback for a degenerate separation.
    """
    query = mesh_query_point_sign(mesh, wp.cw_div(wp.transform_point(X_sw, point), scale), 1.0e6, sign_method)
    surface = wp.cw_mul(wp.mesh_eval_position(mesh, query.face, query.u, query.v), scale)
    velocity = wp.cw_mul(wp.mesh_eval_velocity(mesh, query.face, query.u, query.v), scale)
    face_normal = wp.mesh_eval_face_normal(mesh, query.face)
    return wp.transform_point(X_bs, surface), wp.transform_vector(X_bs, velocity), face_normal


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
    shape_mesh_properties: wp.array[wp.int32],
    params: wp.array[float],
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
    elif family == _MESH_FEATURE_EE_DEPTH:
        sv0 = edge_indices[soft_feature, 2]
        sv1 = edge_indices[soft_feature, 3]
        corners = wp.vec3i(sv0, sv1, -1)
        t = params[tid]
        bary = wp.vec3(1.0 - t, t, 0.0)
        probe = particle_q[sv0] + t * (particle_q[sv1] - particle_q[sv0])
        body_pos, body_vel, face_normal = _nearest_surface_contact(
            mesh, X_bs, X_sw, scale, probe, resolve_mesh_sign_method(shape_mesh_properties[shape_index])
        )
        normal = transform_normal_with_scale(X_ws, scale, face_normal)
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
    contact_count = data.contact_count
    wp.launch(
        _begin_mesh_contacts,
        dim=1,
        inputs=[contacts.soft_contact_count, contact_count],
        device=device,
        record_tape=False,
    )
    if contacts._soft_contact_mesh_features is None:
        contacts._soft_contact_mesh_features = wp.empty(contacts.soft_contact_max, dtype=wp.vec3i, device=device)
    features = contacts._soft_contact_mesh_features
    if contacts._soft_contact_mesh_params is None:
        contacts._soft_contact_mesh_params = wp.empty(contacts.soft_contact_max, dtype=float, device=device)
    params = contacts._soft_contact_mesh_params
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
        model._shape_mesh_properties,
    ]
    parallel_epsilon = detector.edge_edge_parallel_epsilon if detector is not None else 1.0e-5

    if n_vt > 0:
        # CUDA shares the nearest query within a warp. CPU queries keep one
        # worker and the same exact traversal, without redundant sign queries.
        partition_depth = _MESH_QUERY_PARTITION_DEPTH if device.is_cuda else 0
        wp.launch(
            _detect_mesh_vertex_contacts,
            dim=n_vt << partition_depth,
            inputs=[
                vt_pairs,
                partition_depth,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_flags,
                model.shape_scale,
                model.shape_source_ptr,
                model._shape_mesh_properties,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                model.shape_margin,
                gap,
                contact_max,
                *cone_args,
            ],
            outputs=[contact_count, features, contact_shapes],
            device=device,
            record_tape=False,
            block_dim=64,
        )

    if detector is not None and n_tv > 0:
        wp.launch(
            _detect_mesh_face_contacts,
            dim=n_tv,
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
            block_dim=64,
        )

    if detector is not None and n_ee > 0:
        wp.launch(
            _detect_mesh_edge_contacts,
            dim=n_ee,
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
            block_dim=64,
        )

    if detector is not None and model.edge_count and len(data.rigid_faces):
        wp.launch(
            _detect_mesh_edge_penetrations,
            dim=len(data.rigid_faces),
            inputs=[
                data.rigid_faces,
                state.particle_q,
                model.particle_flags,
                model.edge_indices,
                *shape_args,
                model.shape_world,
                detector.bvh_edges.id,
                detector.bvh_edges_group_roots,
                model.world_count,
                contact_max,
            ],
            outputs=[contact_count, features, params, contact_shapes],
            device=device,
            record_tape=False,
            block_dim=64,
        )

    wp.launch(
        _end_mesh_contacts,
        dim=1,
        inputs=[contact_count, contacts.soft_contact_count, contact_max],
        device=device,
        record_tape=False,
    )
    if contact_max == 0:
        return
    wp.launch(
        _evaluate_mesh_contacts,
        dim=contact_max,
        inputs=[
            contacts.soft_contact_count,
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
            model._shape_mesh_properties,
            params,
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

    def __init__(self, model: Model, shape_mask: np.ndarray, vertex_pairs: wp.array[wp.vec2i], gap: float):
        self.vertex_pairs = vertex_pairs
        meshes = _collision_meshes(model, shape_mask)
        if model.tri_count:
            self.rigid_features = _build_rigid_features(model, meshes)
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
        face_rows = [
            np.column_stack((np.full(len(mesh.indices) // 3, shape), np.arange(len(mesh.indices) // 3)))
            for shape, mesh in meshes.items()
        ]
        self.rigid_faces = wp.array(
            np.concatenate(face_rows).astype(np.int32) if face_rows else np.empty((0, 2), np.int32),
            dtype=wp.vec2i,
            device=model.device,
        )
        vertices, vertex_normals, edges, edge_normals = self.rigid_features
        if (
            max(_MESH_QUERY_PARTITIONS * len(vertex_pairs), len(vertices), len(edges), len(self.rigid_faces))
            > np.iinfo(np.int32).max
        ):
            raise ValueError("Mesh contact queries exceed 32-bit indexing capacity.")
        adjacency, concurrent_components = _build_feature_adjacency(
            model, meshes, vertices, edges, gap + model.particle_max_radius
        )
        self.adjacency = [*adjacency, vertex_normals, edge_normals]
        # A query emits at most one contact per feature pair, or two depth probes per
        # face crossing. Bound the wide append counter before allocating; final writes
        # remain capacity checked.
        max_faces = max(len(mesh.indices) // 3 for mesh in meshes.values())
        bound = (
            len(vertex_pairs) * max_faces
            + len(vertices) * model.tri_count
            + len(edges) * model.edge_count
            + 2 * len(self.rigid_faces) * model.edge_count
        )
        if bound > np.iinfo(np.int64).max - np.iinfo(np.int32).max:
            raise ValueError("Mesh contact pairs exceed 64-bit counting capacity.")
        self.contact_count = wp.empty(1, dtype=wp.int64, device=model.device)
        # Size final storage primarily from the deformable workload. Retain a
        # rigid-feature floor for fine flat patches touching very coarse cloth.
        # This remains an estimate; every write checks the caller-sized capacity.
        particle_world = model.particle_world.numpy()
        shape_world = model.shape_world.numpy()[shape_mask]
        shape_counts = np.bincount(shape_world + 1, minlength=model.world_count + 1)

        def count_pairs(indices, column):
            if indices is None:
                return 0
            indices = indices.numpy()
            worlds = particle_world[indices[:, column]]
            counts = np.bincount(worlds + 1, minlength=model.world_count + 1)
            return int(
                counts[0] * len(shape_world)
                + (len(worlds) - counts[0]) * shape_counts[0]
                + counts[1:] @ shape_counts[1:]
            )

        surface_pairs = len(vertex_pairs) + count_pairs(model.tri_indices, 0) + count_pairs(model.edge_indices, 2)
        # Allow four local patches per surface component that a particle can touch at once.
        vertex_pair_shapes = vertex_pairs.numpy()[:, 1]
        vertex_patch_hint = int((4 * concurrent_components[vertex_pair_shapes]).sum(dtype=np.int64))
        self.contact_capacity_hint = max(vertex_patch_hint, surface_pairs, len(vertices) + len(edges))
