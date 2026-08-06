# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative removal of fully inward mesh collision edges."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .types import Mesh

MINVAL = 1.0e-15


def filter_fully_inward_edges(mesh: Mesh, edge_indices: np.ndarray) -> np.ndarray:
    """Drop concave edges whose endpoints have fully inward manifold one-rings.

    A removable edge is shared by exactly two non-degenerate triangles. Both
    endpoint vertices must have connected, closed, consistently oriented
    one-rings, and every one-ring neighbor must lie on the inward side of the
    endpoint's angle-weighted tangent plane. Boundary, non-manifold, saddle,
    flat, and ambiguous features are preserved.

    Args:
        mesh: Source mesh with consistently authored triangle winding.
        edge_indices: Candidate collision-edge vertex pairs.

    Returns:
        A contiguous subset of ``edge_indices`` with fully inward edges removed.
    """
    if len(edge_indices) == 0 or mesh.indices.size == 0 or mesh.vertices.size == 0:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
    canonical = mesh._canonical_vertex_ids()
    canonical_triangles = canonical[triangles]

    # Winding-number SDF construction uses this volume sign to correct a
    # globally inverted mesh. Apply the same correction to the feature tests.
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    signed_volume = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
    diagonal = max(mesh._aabb_diagonal(), 1.0)
    volume_tolerance = np.finfo(np.float64).eps * diagonal**3 * max(len(triangles), 1)
    if abs(signed_volume) <= volume_tolerance:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)
    orientation = 1.0 if signed_volume > 0.0 else -1.0

    orig_edges, _slot_keys, order, keys_sorted, face_normals, face_norms = mesh._build_edge_slot_topology()
    if len(keys_sorted) == 0:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)

    change = np.empty(len(keys_sorted), dtype=bool)
    change[0] = True
    change[1:] = keys_sorted[1:] != keys_sorted[:-1]
    group_starts = np.flatnonzero(change)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = len(keys_sorted)
    group_counts = group_ends - group_starts
    edge_share_count = {
        int(keys_sorted[start]): int(count) for start, count in zip(group_starts, group_counts, strict=True)
    }

    canonical_count = int(canonical.max()) + 1
    canonical_positions = np.empty((canonical_count, 3), dtype=np.float64)
    canonical_positions[canonical] = vertices
    incident: list[list[tuple[int, int]]] = [[] for _ in range(canonical_count)]
    for face_idx, triangle in enumerate(canonical_triangles):
        for local_idx, vertex_idx in enumerate(triangle):
            incident[int(vertex_idx)].append((face_idx, local_idx))

    plane_tolerance = 1.0e-7 * diagonal
    inward_vertices: set[int] = set()
    for vertex_idx, corners in enumerate(incident):
        if len(corners) < 3:
            continue

        link: dict[int, set[int]] = {}
        previous_count: dict[int, int] = {}
        next_count: dict[int, int] = {}
        normal_sum = np.zeros(3, dtype=np.float64)
        topology_valid = True

        for face_idx, local_idx in corners:
            triangle = canonical_triangles[face_idx]
            if len({int(triangle[0]), int(triangle[1]), int(triangle[2])}) != 3:
                topology_valid = False
                break

            previous_vertex = int(triangle[(local_idx - 1) % 3])
            next_vertex = int(triangle[(local_idx + 1) % 3])
            if orientation < 0.0:
                previous_vertex, next_vertex = next_vertex, previous_vertex

            previous_key = (min(vertex_idx, previous_vertex) << 32) | max(vertex_idx, previous_vertex)
            next_key = (min(vertex_idx, next_vertex) << 32) | max(vertex_idx, next_vertex)
            if edge_share_count.get(previous_key) != 2 or edge_share_count.get(next_key) != 2:
                topology_valid = False
                break

            link.setdefault(previous_vertex, set()).add(next_vertex)
            link.setdefault(next_vertex, set()).add(previous_vertex)
            previous_count[previous_vertex] = previous_count.get(previous_vertex, 0) + 1
            next_count[next_vertex] = next_count.get(next_vertex, 0) + 1

            previous_delta = canonical_positions[previous_vertex] - canonical_positions[vertex_idx]
            next_delta = canonical_positions[next_vertex] - canonical_positions[vertex_idx]
            previous_length = float(np.linalg.norm(previous_delta))
            next_length = float(np.linalg.norm(next_delta))
            if previous_length <= MINVAL or next_length <= MINVAL or face_norms[face_idx] <= MINVAL:
                topology_valid = False
                break
            cosine = np.clip(
                np.dot(previous_delta, next_delta) / (previous_length * next_length),
                -1.0,
                1.0,
            )
            corner_angle = float(np.arccos(cosine))
            normal_sum += corner_angle * orientation * face_normals[face_idx] / face_norms[face_idx]

        neighbors = set(link)
        if not topology_valid or len(neighbors) < 3:
            continue
        if any(
            len(link[neighbor]) != 2 or previous_count.get(neighbor) != 1 or next_count.get(neighbor) != 1
            for neighbor in neighbors
        ):
            continue

        reached: set[int] = set()
        pending = [next(iter(neighbors))]
        while pending:
            neighbor = pending.pop()
            if neighbor in reached:
                continue
            reached.add(neighbor)
            pending.extend(link[neighbor] - reached)
        if reached != neighbors:
            continue

        normal_length = float(np.linalg.norm(normal_sum))
        if normal_length <= MINVAL:
            continue
        normal = normal_sum / normal_length
        neighbor_indices = np.fromiter(neighbors, dtype=np.int64)
        neighbor_offsets = canonical_positions[neighbor_indices] - canonical_positions[vertex_idx]
        heights = neighbor_offsets @ normal
        if float(np.min(heights)) >= -plane_tolerance and float(np.max(heights)) > plane_tolerance:
            inward_vertices.add(vertex_idx)

    if len(inward_vertices) < 2:
        return np.ascontiguousarray(edge_indices, dtype=np.int32)

    concave_edge_keys: set[int] = set()
    for group_idx in np.flatnonzero(group_counts == 2):
        start = group_starts[group_idx]
        slots = order[start : start + 2]
        tri_a = int(slots[0] // 3)
        tri_b = int(slots[1] // 3)
        if face_norms[tri_a] <= MINVAL or face_norms[tri_b] <= MINVAL:
            continue

        edge = orig_edges[slots[0]]
        canonical_a, canonical_b = sorted((int(canonical[edge[0]]), int(canonical[edge[1]])))
        opposite_a = [
            int(vertex) for vertex in triangles[tri_a] if int(canonical[vertex]) not in (canonical_a, canonical_b)
        ]
        opposite_b = [
            int(vertex) for vertex in triangles[tri_b] if int(canonical[vertex]) not in (canonical_a, canonical_b)
        ]
        if len(opposite_a) != 1 or len(opposite_b) != 1:
            continue

        edge_point = vertices[edge[0]]
        normal_a = orientation * face_normals[tri_a] / face_norms[tri_a]
        normal_b = orientation * face_normals[tri_b] / face_norms[tri_b]
        side_b = float(np.dot(normal_a, vertices[opposite_b[0]] - edge_point))
        side_a = float(np.dot(normal_b, vertices[opposite_a[0]] - edge_point))
        if side_a > plane_tolerance and side_b > plane_tolerance:
            concave_edge_keys.add(int(keys_sorted[start]))

    keep = np.ones(len(edge_indices), dtype=bool)
    for edge_idx, edge in enumerate(edge_indices):
        canonical_a, canonical_b = sorted((int(canonical[edge[0]]), int(canonical[edge[1]])))
        key = (canonical_a << 32) | canonical_b
        if canonical_a in inward_vertices and canonical_b in inward_vertices and key in concave_edge_keys:
            keep[edge_idx] = False
    return np.ascontiguousarray(edge_indices[keep], dtype=np.int32)
