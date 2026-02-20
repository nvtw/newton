# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Smooth normal generation with sharp-edge detection.

1:1 port of the C# ``MinimalDlssRRViewer.Normals`` algorithm.

Pipeline:
    1. De-duplicate vertices (exact float equality).
    2. Map triangles to canonical indices, remove degenerates.
    3. Remove duplicate triangles (ignoring winding).
    4. Make triangle orientation consistent via BFS flood-fill.
    5. Build edge list and triangle adjacency.
    6. For each vertex, sort adjacent triangles in fan order using
       the ``HashSegmentConnector`` strip-builder.
    7. Walk the fan, accumulate angle-weighted normals, and split at
       sharp edges (angle > threshold).

The public entry point is :func:`compute_normals`.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 1. Duplicate point removal
# ---------------------------------------------------------------------------

def _duplicate_map(points: NDArray[np.float32]) -> NDArray[np.int32]:
    """Map each vertex index to its canonical (first-occurrence) index.

    Uses a dict keyed on the raw float triple for exact equality, matching
    the C# ``DuplicatePointRemover.DuplicateMap``.
    """
    n = len(points)
    canon = np.arange(n, dtype=np.int32)
    seen: dict[tuple[float, float, float], int] = {}
    for i in range(n):
        key = (float(points[i, 0]), float(points[i, 1]), float(points[i, 2]))
        if key in seen:
            canon[i] = seen[key]
        else:
            seen[key] = i
    return canon


def _map_triangles(
    tris: NDArray[np.int32],
    canon: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Remap triangle indices through *canon* and drop degenerates.

    Returns ``(mapped_tris, keep_mask)`` where *keep_mask* is a boolean
    array indicating which original triangles survived.
    """
    mapped = canon[tris]
    keep = ~(
        (mapped[:, 0] == mapped[:, 1])
        | (mapped[:, 1] == mapped[:, 2])
        | (mapped[:, 0] == mapped[:, 2])
    )
    return mapped[keep], keep


# ---------------------------------------------------------------------------
# 2. Duplicate triangle removal + consistent orientation
# ---------------------------------------------------------------------------

def _remove_duplicate_triangles(
    mapped: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Remove duplicate triangles (ignoring winding order).

    Matches C# ``DuplicateTriangleRemover.RemoveDuplicatedTriangles``.
    """
    sorted_idx = np.sort(mapped, axis=1)
    _, unique_idx = np.unique(sorted_idx, axis=0, return_index=True)
    unique_idx.sort()
    return mapped[unique_idx]


def _make_orientation_consistent(tris: NDArray[np.int32]) -> NDArray[np.int32]:
    """BFS flood-fill to make triangle winding consistent.

    Matches C# ``DuplicateTriangleRemover.MakeTriOrientationConsistent``.
    """
    n = len(tris)
    if n == 0:
        return tris

    adj = _build_adjacency(tris)
    done = np.zeros(n, dtype=np.bool_)
    flip = np.zeros(n, dtype=np.bool_)

    neighbor_edges = ((0, 1), (1, 2), (2, 0))
    seed = 0
    stack: list[int] = []

    while True:
        if not stack:
            while seed < n and done[seed]:
                seed += 1
            if seed == n:
                break
            done[seed] = True
            flip[seed] = False
            stack.append(seed)

        idx = stack.pop()
        f = flip[idx]
        tri = tris[idx]
        neighbors = (adj[idx, 0], adj[idx, 1], adj[idx, 2])

        for i in range(3):
            ni = neighbors[i]
            if ni >= 0 and not done[ni]:
                done[ni] = True
                ntri = tris[ni]
                v0 = tri[neighbor_edges[i][0]]
                v1 = tri[neighbor_edges[i][1]]
                j = _index_of(ntri, v0)
                flip[ni] = (ntri[(j + 1) % 3] == v1) != f
                stack.append(ni)

    result = tris.copy()
    to_flip = flip
    result[to_flip] = result[to_flip][:, [1, 0, 2]]
    return result


# ---------------------------------------------------------------------------
# 3. Edge list + adjacency
# ---------------------------------------------------------------------------

def _build_edge_list(tris: NDArray[np.int32]) -> dict[tuple[int, int], list[int]]:
    """Build a mapping from canonical edge ``(min, max)`` to triangle indices.

    Edges shared by >2 triangles are marked invalid (empty list).
    Matches C# ``Adjacency.BuildEdgeList``.
    """
    edges: dict[tuple[int, int], list[int]] = {}
    n = len(tris)
    edge_pairs = ((0, 1), (1, 2), (2, 0))
    for i in range(n):
        a, b, c = int(tris[i, 0]), int(tris[i, 1]), int(tris[i, 2])
        verts = (a, b, c)
        for e0, e1 in edge_pairs:
            v0, v1 = verts[e0], verts[e1]
            key = (min(v0, v1), max(v0, v1))
            if key in edges:
                lst = edges[key]
                if len(lst) == 0:
                    pass  # already invalid
                elif len(lst) == 1:
                    lst.append(i)
                else:
                    lst.clear()  # >2 triangles -> invalid
            else:
                edges[key] = [i]
    return edges


def _build_adjacency(tris: NDArray[np.int32]) -> NDArray[np.int32]:
    """Build per-triangle adjacency array ``(N, 3)`` with neighbour indices.

    ``adj[i, 0]`` = neighbour across edge AB, ``[i,1]`` = BC, ``[i,2]`` = CA.
    ``-1`` means no neighbour.  Matches C# ``Adjacency.BuildAdjacencyInformation``.
    """
    n = len(tris)
    adj = np.full((n, 3), -1, dtype=np.int32)
    edges = _build_edge_list(tris)

    for lst in edges.values():
        if len(lst) != 2:
            continue
        i0, i1 = lst
        t0 = tris[i0]
        t1 = tris[i1]
        _set_neighbour(adj, i0, t0, t1, i1)
        _set_neighbour(adj, i1, t1, t0, i0)
    return adj


def _set_neighbour(
    adj: NDArray[np.int32],
    tri_idx: int,
    tri: NDArray[np.int32],
    other: NDArray[np.int32],
    other_idx: int,
) -> None:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    os = {int(other[0]), int(other[1]), int(other[2])}
    if b in os and c in os:
        adj[tri_idx, 1] = other_idx
    elif c in os and a in os:
        adj[tri_idx, 2] = other_idx
    elif a in os and b in os:
        adj[tri_idx, 0] = other_idx


# ---------------------------------------------------------------------------
# 4. HashSegmentConnector
# ---------------------------------------------------------------------------

def _connect_segments(
    segments: list[tuple[int, int]],
) -> list[tuple[list[int], bool]]:
    """Connect edge-segments into strips/loops.

    Returns a list of ``(vertex_strip, is_closed)`` tuples.
    Matches C# ``HashSegmentConnector.ConnectAndResolve``.
    """
    if not segments:
        raise ValueError("No segments to connect")

    if len(segments) == 1:
        s, e = segments[0]
        if s == e:
            raise ValueError("Single segment is degenerate")
        return [([s, e], False)]

    # Build bidirectional adjacency map.
    link_map: dict[int, list[int]] = {}
    processed: set[int] = set()

    for s, e in segments:
        if s == e:
            continue
        if s not in link_map:
            link_map[s] = []
        if e not in link_map[s]:
            link_map[s].append(e)
        if e not in link_map:
            link_map[e] = []
        if s not in link_map[e]:
            link_map[e].append(s)

    results: list[tuple[list[int], bool]] = []

    for start_key, start_neighbors in link_map.items():
        if start_key in processed:
            continue

        strip_a = [start_key]
        processed.add(start_key)
        last_valid = start_key
        neighbors = start_neighbors

        if neighbors:
            stack = [neighbors[0]]
            while stack:
                target = stack.pop()
                if target in processed:
                    continue
                last_valid = target
                processed.add(target)
                strip_a.append(target)
                for nb in link_map.get(target, ()):
                    stack.append(nb)

            closed = False
            if len(strip_a) > 2:
                closed = start_key in link_map.get(last_valid, ())

        else:
            closed = False

        # Backward walk from link_map[start_key][1] if it exists
        strip_b: list[int] = []
        if len(neighbors) > 1:
            stack = [neighbors[1]]
            while stack:
                target = stack.pop()
                if target in processed:
                    continue
                processed.add(target)
                strip_b.append(target)
                for nb in link_map.get(target, ()):
                    stack.append(nb)

        strip_a.reverse()
        strip_a.extend(strip_b)
        results.append((strip_a, closed))

    return results


# ---------------------------------------------------------------------------
# 5. AutoNormals main algorithm
# ---------------------------------------------------------------------------

def _index_of(tri: NDArray[np.int32], v: int) -> int:
    if tri[0] == v:
        return 0
    if tri[1] == v:
        return 1
    if tri[2] == v:
        return 2
    return -1


def _get_remaining(tri: NDArray[np.int32], v: int) -> tuple[int, int]:
    a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
    if v == a:
        return (b, c)
    if v == b:
        return (c, a)
    if v == c:
        return (a, b)
    raise ValueError("Vertex not in triangle")


def _angle_between(l: NDArray[np.float32], r: NDArray[np.float32]) -> float:
    denom = math.sqrt(float(np.dot(l, l)) * float(np.dot(r, r)))
    if denom < 1e-30:
        return 0.0
    d = float(np.dot(l, r)) / denom
    if d <= -1.0:
        return math.pi
    if d >= 1.0:
        return 0.0
    return math.acos(d)


def _tri_angle_at_vertex(
    points: NDArray[np.float32],
    tri: NDArray[np.int32],
    v: int,
) -> float:
    rem = _get_remaining(tri, v)
    p = points[v]
    return _angle_between(points[rem[0]] - p, points[rem[1]] - p)


def _sort_fan(
    points: NDArray[np.float32],
    vertex: int,
    tri_indices: list[int],
    tris: NDArray[np.int32],
) -> list[tuple[list[int], bool]]:
    """Sort adjacent triangles into fan order around *vertex*.

    Returns a list of ``(sorted_tri_indices, is_closed)`` per connected
    component.  Matches C# ``SortTrianglesInFanOrder``.
    """
    edge_to_tri: dict[tuple[int, int], int] = {}
    segments: list[tuple[int, int]] = []

    for ti in tri_indices:
        rem = _get_remaining(tris[ti], vertex)
        edge_to_tri[rem] = ti
        segments.append(rem)

    # Detect non-manifold vertices: if a remaining-edge endpoint appears
    # in >2 segments, replace each occurrence with a unique synthetic ID.
    degree: dict[int, int] = {}
    for s, e in segments:
        degree[s] = degree.get(s, 0) + 1
        degree[e] = degree.get(e, 0) + 1

    synthetic = -(1 << 30)
    new_segments: list[tuple[int, int]] = []
    new_edge_to_tri: dict[tuple[int, int], int] = {}
    for i, (sx, sy) in enumerate(segments):
        if degree[sx] > 2:
            synthetic += 1
            nx = synthetic
        else:
            nx = sx
        if degree[sy] > 2:
            synthetic += 1
            ny = synthetic
        else:
            ny = sy
        new_seg = (nx, ny)
        new_edge_to_tri[new_seg] = edge_to_tri[segments[i]]
        new_segments.append(new_seg)

    parts = _connect_segments(new_segments)

    results: list[tuple[list[int], bool]] = []
    for strip, closed in parts:
        vertices = [*strip, strip[0]] if closed else strip
        sorted_tris: list[int] = []
        for j in range(1, len(vertices)):
            edge = (vertices[j - 1], vertices[j])
            tid = new_edge_to_tri.get(edge)
            if tid is None:
                edge = (edge[1], edge[0])
                tid = new_edge_to_tri.get(edge)
                if tid is None:
                    raise RuntimeError("Edge not found in fan sort")
            sorted_tris.append(tid)
        results.append((sorted_tris, closed))

    return results


def _compute_vertex_normals(
    points: NDArray[np.float32],
    vertex: int,
    adj_tris: list[int],
    closed_loop: bool,
    tris: NDArray[np.int32],
    tri_normals: NDArray[np.float32],
    adj: NDArray[np.int32],
    angle_threshold: float,
    result: NDArray[np.float32],
) -> None:
    """Accumulate angle-weighted normals for *vertex*, splitting at sharp edges.

    Writes directly into *result* which has shape ``(num_tris, 3, 3)``.
    Matches C# ``ComputeVertexNormals``.
    """
    num = len(adj_tris)

    has_sharp = [False] * num
    start = 0
    for i in range(num):
        ti = adj_tris[i]
        ni = adj_tris[(i + 1) % num]
        # Check if triangles share an edge via adjacency
        if adj[ti, 0] == ni or adj[ti, 1] == ni or adj[ti, 2] == ni:
            angle = _angle_between(tri_normals[ti], tri_normals[ni])
            if angle > angle_threshold:
                has_sharp[i] = True
                if start == 0:
                    start = (i + 1) % num
        else:
            has_sharp[i] = True
            if start == 0:
                start = (i + 1) % num

    if not closed_loop:
        start = 0
        end = num
        has_sharp[num - 1] = True
    else:
        end = num + start

    smoothed = np.zeros(3, dtype=np.float64)
    first = start
    for k in range(start, end):
        i = k % num
        ti = adj_tris[i]
        w = _tri_angle_at_vertex(points, tris[ti], vertex)
        smoothed += w * tri_normals[ti].astype(np.float64)

        if has_sharp[i]:
            norm = np.linalg.norm(smoothed)
            if norm > 1e-30:
                smoothed /= norm
            sn = smoothed.astype(np.float32)
            for j in range(first, k + 1):
                ti2 = adj_tris[j % num]
                local_id = _index_of(tris[ti2], vertex)
                result[ti2, local_id] = sn
            smoothed[:] = 0.0
            first = k + 1

    norm = np.linalg.norm(smoothed)
    if norm > 1e-30:
        smoothed /= norm
    sn = smoothed.astype(np.float32)
    for j in range(first, end):
        ti2 = adj_tris[j % num]
        local_id = _index_of(tris[ti2], vertex)
        result[ti2, local_id] = sn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_normals(
    points: NDArray[np.float32],
    triangles: NDArray[np.int32],
    angle_threshold_radians: float = math.radians(60.0),
) -> NDArray[np.float32]:
    """Compute smooth normals with sharp-edge detection.

    Args:
        points: Vertex positions, shape ``(V, 3)``, float32.
        triangles: Triangle indices, shape ``(T, 3)``, int32.
        angle_threshold_radians: Dihedral angle above which an edge is
            considered sharp [rad].

    Returns:
        Per-corner normals, shape ``(T, 3, 3)`` — ``result[t, c]`` is the
        normal at corner *c* of triangle *t*.  The triangle indexing
        corresponds to the *output* triangle list (after degenerate /
        duplicate removal and orientation fix).
    """
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    triangles = np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1, 3)

    # 1. De-duplicate vertices
    canon = _duplicate_map(points)
    mapped, _keep = _map_triangles(triangles, canon)
    if len(mapped) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    # 2. Remove duplicate triangles + fix orientation
    mapped = _remove_duplicate_triangles(mapped)
    mapped = _make_orientation_consistent(mapped)

    num_tris = len(mapped)

    # 3. Triangle normals
    a = points[mapped[:, 0]]
    b = points[mapped[:, 1]]
    c = points[mapped[:, 2]]
    raw = np.cross(b - a, c - a)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    tri_normals = (raw / norms).astype(np.float32)

    # 4. Vertex -> triangle adjacency
    vert_to_tris: dict[int, list[int]] = {}
    for i in range(num_tris):
        for v in (int(mapped[i, 0]), int(mapped[i, 1]), int(mapped[i, 2])):
            if v not in vert_to_tris:
                vert_to_tris[v] = []
            vert_to_tris[v].append(i)

    # 5. Triangle adjacency
    adj = _build_adjacency(mapped)

    # 6. Output buffer
    result = np.zeros((num_tris, 3, 3), dtype=np.float32)

    # 7. Per-vertex fan traversal
    for vertex, adj_list in vert_to_tris.items():
        if not adj_list:
            continue
        fans = _sort_fan(points, vertex, adj_list, mapped)
        for sorted_tris, closed in fans:
            _compute_vertex_normals(
                points,
                vertex,
                sorted_tris,
                closed,
                mapped,
                tri_normals,
                adj,
                angle_threshold_radians,
                result,
            )

    return result
