# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CPU semantic reference: deterministic connected contact-history groups.

Connectivity is supplied explicitly, not inferred from a whole body pair or a
hidden distance. This is not a production GPU adapter. Point IDs order contacts;
they do NOT certify material-point persistence across generations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Groups:
    """Active-length CSR and maps directly compatible with history input layout."""

    counts: np.ndarray
    offsets: np.ndarray
    members: np.ndarray
    point_patch: np.ndarray
    representatives: np.ndarray


def group_contacts(keys, normals, point_ids, edges, *, normal_cosine):
    """Greedy deterministic seed-cone connected components of explicit graph.

    Choose the smallest (world,bodies,material,stable-ID) unassigned point.
    Grow only through supplied graph edges and unassigned points compatible
    with that seed key/normal. Repeat. Every resulting patch is connected and
    within its representative's normal cone; transitive normal chains cannot
    rotate the representative. Group/member ordering is independent of input
    row/edge ordering when stable IDs and input geometry are unchanged.
    """
    keys = np.asarray(keys)
    normals = np.asarray(normals)
    point_ids = np.asarray(point_ids)
    edges = np.asarray(edges)
    n = len(keys)
    if keys.shape != (n, 4) or normals.shape != (n, 3) or point_ids.shape != (n,):
        raise ValueError("Expected keys[n,4], normals[n,3], point_ids[n]")
    if normals.dtype != np.float32 or keys.dtype != np.int32 or point_ids.dtype != np.int64:
        raise ValueError("Use FP32 normals, int32 keys and int64 stable ordering IDs")
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Connectivity must be explicit integer edges[m,2]")
    if not np.isfinite(normal_cosine) or not 0 <= normal_cosine <= 1:
        raise ValueError("normal_cosine must lie in [0,1]")
    norm2 = np.sum(normals * normals, axis=1)
    if not np.isfinite(normals).all() or np.any(abs(norm2 - np.float32(1)) > 32 * np.finfo(np.float32).eps):
        raise ValueError("Normals must be finite unit vectors; no implicit normalization")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Connectivity endpoint outside current contact range")
    order_keys = [(*map(int, keys[i]), int(point_ids[i])) for i in range(n)]
    if len(set(order_keys)) != n:
        raise ValueError("Ordering IDs must be unique within each compatibility key")
    order = sorted(range(n), key=order_keys.__getitem__)
    adjacency = [set() for _ in range(n)]
    for a, b in edges:
        if a != b:
            adjacency[a].add(int(b))
            adjacency[b].add(int(a))
    assignment = np.full(n, -1, np.int32)
    groups = []
    representatives = []
    for seed in order:
        if assignment[seed] >= 0:
            continue
        group_id = len(groups)
        assignment[seed] = group_id
        pending = [seed]
        group = []
        while pending:
            point = pending.pop()
            group.append(point)
            for other in sorted(adjacency[point], key=order_keys.__getitem__):
                if assignment[other] >= 0 or not np.array_equal(keys[other], keys[seed]):
                    continue
                if np.dot(normals[seed], normals[other]) < np.float32(normal_cosine):
                    continue
                assignment[other] = group_id
                pending.append(other)
        groups.append(sorted(group, key=order_keys.__getitem__))
        representatives.append(seed)
    offsets = np.cumsum([0, *map(len, groups)], dtype=np.int32)
    members = np.array([point for group in groups for point in group], np.int32)
    assert len(members) == n and len(np.unique(members)) == n
    return Groups(
        np.array([len(groups), n], np.int32), offsets, members, assignment, np.array(representatives, np.int32)
    )
