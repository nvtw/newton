# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-first prototype that flags mesh edges as candidates for removal.

For every manifold edge of a triangle mesh (an edge shared by exactly two
triangles, with a finite dihedral angle), we build an oriented bounding box
in the local edge frame:

- ``dir`` along the edge (length covers the edge plus a small overhang),
- ``tang`` in the average tangent plane (perpendicular to the edge, in plane),
- ``normal`` along the average of the two adjacent face normals.

We then run the SAP broad phase (:class:`BroadPhaseSAP`) to obtain candidate
overlapping AABB pairs and, for every pair ``(a, b)``, exactly test whether
edge ``a``'s box fully contains edge ``b``'s segment (both endpoints inside the
oriented box). When ``a`` swallows ``b``, we mark ``b`` as a removal candidate.
Mutual containment marks both edges so a follow-up resolution step can choose
which one to actually drop without creating cyclic dependencies.

The output includes a CSR-style adjacency listing the swallowed edges per box
plus per-edge counts. All heavy lifting happens in Warp kernels — NumPy is used
only at the boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .broad_phase_sap import BroadPhaseSAP
from .flags import ShapeFlags

# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class EdgeRedundancyResult:
    """Per-edge containment results returned by :func:`find_redundant_edges`.

    Indices are local to ``edge_indices`` (i.e. the manifold-edge subset that
    actually participated, *not* the original :attr:`Mesh.edges` rows).

    Attributes:
        edge_indices [-]: Manifold edge vertex pairs ``(M, 2)`` (subset of
            :attr:`Mesh.edges`).
        dihedral_angles [rad]: Per-edge dihedral angles, length ``M``.
        candidate_for_removal [-]: Boolean mask, length ``M``. ``True`` for edges
            whose segment is fully contained by at least one other edge's box.
        num_containers_per_edge [-]: For each edge, how many distinct boxes
            fully contain it. ``0`` means kept; ``>= 1`` means a removal
            candidate.
        swallow_count_per_box [-]: For each box, how many other edges it fully
            contains. Useful for picking a winner in a follow-up resolution
            step.
        swallowed_offsets [-]: CSR offsets, length ``M + 1``.
        swallowed_indices [-]: CSR values, length ``swallowed_offsets[-1]``.
            Edges contained in box ``j`` are
            ``swallowed_indices[swallowed_offsets[j]:swallowed_offsets[j+1]]``.
        broad_phase_pair_count [-]: Number of overlapping AABB pairs returned by
            the SAP broad phase before exact containment filtering.
        aabb_diagonal [m]: Diagonal of the mesh's world-space AABB.
        half_height [m]: Resolved value used for the box half-extent along the
            edge normal.
        half_width [m]: Resolved value used for both the edge tangent half-
            extent and the per-end overhang along the edge direction.
    """

    edge_indices: np.ndarray
    dihedral_angles: np.ndarray
    candidate_for_removal: np.ndarray
    num_containers_per_edge: np.ndarray
    swallow_count_per_box: np.ndarray
    swallowed_offsets: np.ndarray
    swallowed_indices: np.ndarray
    broad_phase_pair_count: int
    aabb_diagonal: float
    half_height: float
    half_width: float


# -----------------------------------------------------------------------------
# Warp kernels
# -----------------------------------------------------------------------------


@wp.kernel
def _build_edge_box_kernel(
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    avg_normals: wp.array[wp.vec3],
    half_height: float,
    half_width: float,
    # Outputs
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
):
    i = wp.tid()

    e = edge_indices[i]
    v0 = vertices[e[0]]
    v1 = vertices[e[1]]

    edge_vec = v1 - v0
    edge_len = wp.length(edge_vec)
    n = avg_normals[i]
    n_len = wp.length(n)

    # Degenerate edge or missing normal (NaN-filled) -> mark as no-box.
    if edge_len <= 1.0e-12 or n_len <= 1.0e-12 or wp.isnan(n[0]):
        box_center[i] = wp.vec3(0.0, 0.0, 0.0)
        box_axis_dir[i] = wp.vec3(1.0, 0.0, 0.0)
        box_axis_tang[i] = wp.vec3(0.0, 1.0, 0.0)
        box_axis_normal[i] = wp.vec3(0.0, 0.0, 1.0)
        box_half_extents[i] = wp.vec3(0.0, 0.0, 0.0)
        box_valid[i] = 0
        return

    dir_e = edge_vec / edge_len
    n_unit = n / n_len

    tang = wp.cross(n_unit, dir_e)
    tang_len = wp.length(tang)
    if tang_len <= 1.0e-12:
        box_center[i] = wp.vec3(0.0, 0.0, 0.0)
        box_axis_dir[i] = wp.vec3(1.0, 0.0, 0.0)
        box_axis_tang[i] = wp.vec3(0.0, 1.0, 0.0)
        box_axis_normal[i] = wp.vec3(0.0, 0.0, 1.0)
        box_half_extents[i] = wp.vec3(0.0, 0.0, 0.0)
        box_valid[i] = 0
        return

    tang = tang / tang_len
    # Re-orthogonalize the normal so the frame stays orthonormal even if
    # avg_normal is not exactly perpendicular to dir_e.
    normal = wp.cross(dir_e, tang)

    box_center[i] = 0.5 * (v0 + v1)
    box_axis_dir[i] = dir_e
    box_axis_tang[i] = tang
    box_axis_normal[i] = normal
    box_half_extents[i] = wp.vec3(0.5 * edge_len + half_width, half_width, half_height)
    box_valid[i] = 1


@wp.kernel
def _compute_box_aabb_kernel(
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    # Outputs
    aabb_lower: wp.array[wp.vec3],
    aabb_upper: wp.array[wp.vec3],
):
    i = wp.tid()

    if box_valid[i] == 0:
        # Degenerate box collapsed to a single point so it is never overlapped.
        aabb_lower[i] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
        aabb_upper[i] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
        return

    c = box_center[i]
    h = box_half_extents[i]
    rdir = box_axis_dir[i]
    rtan = box_axis_tang[i]
    rnor = box_axis_normal[i]

    # World half-extents via |R| * h (R has axes as columns).
    hx = wp.abs(rdir[0]) * h[0] + wp.abs(rtan[0]) * h[1] + wp.abs(rnor[0]) * h[2]
    hy = wp.abs(rdir[1]) * h[0] + wp.abs(rtan[1]) * h[1] + wp.abs(rnor[1]) * h[2]
    hz = wp.abs(rdir[2]) * h[0] + wp.abs(rtan[2]) * h[1] + wp.abs(rnor[2]) * h[2]
    world_half = wp.vec3(hx, hy, hz)

    aabb_lower[i] = c - world_half
    aabb_upper[i] = c + world_half


@wp.func
def _box_contains_point(
    p: wp.vec3,
    center: wp.vec3,
    axis_dir: wp.vec3,
    axis_tang: wp.vec3,
    axis_normal: wp.vec3,
    half_extents: wp.vec3,
    eps: float,
) -> int:
    d = p - center
    pd = wp.dot(d, axis_dir)
    pt = wp.dot(d, axis_tang)
    pn = wp.dot(d, axis_normal)
    inside = int(0)
    if (
        wp.abs(pd) <= half_extents[0] + eps
        and wp.abs(pt) <= half_extents[1] + eps
        and wp.abs(pn) <= half_extents[2] + eps
    ):
        inside = 1
    return inside


@wp.func
def _box_contains_edge(
    edge_idx: int,
    box_idx: int,
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    eps: float,
) -> int:
    if box_valid[box_idx] == 0:
        return 0
    e = edge_indices[edge_idx]
    v0 = vertices[e[0]]
    v1 = vertices[e[1]]
    c = box_center[box_idx]
    rdir = box_axis_dir[box_idx]
    rtan = box_axis_tang[box_idx]
    rnor = box_axis_normal[box_idx]
    h = box_half_extents[box_idx]
    in0 = _box_contains_point(v0, c, rdir, rtan, rnor, h, eps)
    in1 = _box_contains_point(v1, c, rdir, rtan, rnor, h, eps)
    return in0 * in1


@wp.kernel
def _count_swallowed_per_box_kernel(
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    eps: float,
    # In/out
    swallow_count_per_box: wp.array[wp.int32],
    num_containers_per_edge: wp.array[wp.int32],
):
    pid = wp.tid()
    if pid >= candidate_pair_count[0]:
        return
    pair = candidate_pair[pid]
    a = pair[0]
    b = pair[1]
    if a == b:
        return

    contains_a_b = _box_contains_edge(
        b,
        a,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    contains_b_a = _box_contains_edge(
        a,
        b,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    if contains_a_b == 1:
        wp.atomic_add(swallow_count_per_box, a, 1)
        wp.atomic_add(num_containers_per_edge, b, 1)
    if contains_b_a == 1:
        wp.atomic_add(swallow_count_per_box, b, 1)
        wp.atomic_add(num_containers_per_edge, a, 1)


@wp.kernel
def _scatter_swallowed_per_box_kernel(
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    box_center: wp.array[wp.vec3],
    box_axis_dir: wp.array[wp.vec3],
    box_axis_tang: wp.array[wp.vec3],
    box_axis_normal: wp.array[wp.vec3],
    box_half_extents: wp.array[wp.vec3],
    box_valid: wp.array[wp.int32],
    swallowed_offsets: wp.array[wp.int32],
    eps: float,
    # In/out
    write_cursor: wp.array[wp.int32],
    swallowed_indices: wp.array[wp.int32],
):
    pid = wp.tid()
    if pid >= candidate_pair_count[0]:
        return
    pair = candidate_pair[pid]
    a = pair[0]
    b = pair[1]
    if a == b:
        return

    contains_a_b = _box_contains_edge(
        b,
        a,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    contains_b_a = _box_contains_edge(
        a,
        b,
        vertices,
        edge_indices,
        box_center,
        box_axis_dir,
        box_axis_tang,
        box_axis_normal,
        box_half_extents,
        box_valid,
        eps,
    )
    if contains_a_b == 1:
        slot = wp.atomic_add(write_cursor, a, 1)
        swallowed_indices[swallowed_offsets[a] + slot] = b
    if contains_b_a == 1:
        slot = wp.atomic_add(write_cursor, b, 1)
        swallowed_indices[swallowed_offsets[b] + slot] = a


@wp.kernel
def _mark_candidates_kernel(
    num_containers_per_edge: wp.array[wp.int32],
    candidate_for_removal: wp.array[wp.int32],
):
    i = wp.tid()
    if num_containers_per_edge[i] > 0:
        candidate_for_removal[i] = 1
    else:
        candidate_for_removal[i] = 0


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def find_redundant_edges(
    mesh,
    *,
    half_height: float | None = None,
    half_width: float | None = None,
    angle_threshold_rad: float = 0.0,
    initial_pair_capacity_factor: int = 8,
    max_retries: int = 3,
    device=None,
) -> EdgeRedundancyResult:
    """Flag manifold edges of *mesh* whose neighborhood is fully covered by
    another edge's oriented bounding box.

    Args:
        mesh: A :class:`newton.Mesh` instance.
        half_height [m]: Box half-extent along the edge normal. Defaults to
            ``1e-3 * D`` where ``D`` is the mesh AABB diagonal.
        half_width [m]: Box half-extent along the in-plane tangent (and the
            per-end overhang along the edge direction). Defaults to ``5e-3 * D``.
        angle_threshold_rad [rad]: Only edges with dihedral angle greater than
            or equal to this threshold are considered. ``0`` (default) keeps
            every manifold edge.
        initial_pair_capacity_factor: Initial broad-phase pair-buffer size in
            multiples of the manifold-edge count.
        max_retries: How many times to grow the pair buffer on overflow.
        device: Optional Warp device. ``None`` picks the current default.

    Returns:
        :class:`EdgeRedundancyResult` containing per-edge containment data.
    """
    edges_np, angles_np, normals_np = mesh._filter_edges_by_dihedral_angle(angle_threshold_rad, return_diagnostics=True)

    # Manifold edges are exactly the ones with finite per-edge diagnostics.
    manifold_mask = np.isfinite(angles_np) & np.all(np.isfinite(normals_np), axis=1)
    edge_indices_np = edges_np[manifold_mask].astype(np.int32, copy=False)
    edge_angles_np = angles_np[manifold_mask].astype(np.float32, copy=False)
    edge_normals_np = normals_np[manifold_mask].astype(np.float32, copy=False)
    n_edges = int(len(edge_indices_np))

    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)

    aabb_min = vertices_np.min(axis=0) if len(vertices_np) > 0 else np.zeros(3, dtype=np.float32)
    aabb_max = vertices_np.max(axis=0) if len(vertices_np) > 0 else np.zeros(3, dtype=np.float32)
    diagonal = float(np.linalg.norm(aabb_max - aabb_min))

    resolved_half_height = float(half_height) if half_height is not None else 1.0e-3 * diagonal
    resolved_half_width = float(half_width) if half_width is not None else 5.0e-3 * diagonal

    # Empty / degenerate fast paths.
    if n_edges == 0:
        empty_offsets = np.zeros(1, dtype=np.int32)
        empty_indices = np.zeros(0, dtype=np.int32)
        empty_int = np.zeros(0, dtype=np.int32)
        empty_bool = np.zeros(0, dtype=bool)
        return EdgeRedundancyResult(
            edge_indices=edge_indices_np.reshape(0, 2),
            dihedral_angles=edge_angles_np,
            candidate_for_removal=empty_bool,
            num_containers_per_edge=empty_int,
            swallow_count_per_box=empty_int,
            swallowed_offsets=empty_offsets,
            swallowed_indices=empty_indices,
            broad_phase_pair_count=0,
            aabb_diagonal=diagonal,
            half_height=resolved_half_height,
            half_width=resolved_half_width,
        )

    if device is None:
        device = wp.get_preferred_device()

    with wp.ScopedDevice(device):
        vertices_wp = wp.array(vertices_np, dtype=wp.vec3)
        edge_indices_wp = wp.array(edge_indices_np.reshape(-1, 2), dtype=wp.vec2i)
        avg_normals_wp = wp.array(edge_normals_np.reshape(-1, 3), dtype=wp.vec3)

        box_center = wp.empty(n_edges, dtype=wp.vec3)
        box_axis_dir = wp.empty(n_edges, dtype=wp.vec3)
        box_axis_tang = wp.empty(n_edges, dtype=wp.vec3)
        box_axis_normal = wp.empty(n_edges, dtype=wp.vec3)
        box_half_extents = wp.empty(n_edges, dtype=wp.vec3)
        box_valid = wp.zeros(n_edges, dtype=wp.int32)

        wp.launch(
            kernel=_build_edge_box_kernel,
            dim=n_edges,
            inputs=[
                vertices_wp,
                edge_indices_wp,
                avg_normals_wp,
                resolved_half_height,
                resolved_half_width,
            ],
            outputs=[
                box_center,
                box_axis_dir,
                box_axis_tang,
                box_axis_normal,
                box_half_extents,
                box_valid,
            ],
        )

        aabb_lower = wp.empty(n_edges, dtype=wp.vec3)
        aabb_upper = wp.empty(n_edges, dtype=wp.vec3)
        wp.launch(
            kernel=_compute_box_aabb_kernel,
            dim=n_edges,
            inputs=[
                box_center,
                box_axis_dir,
                box_axis_tang,
                box_axis_normal,
                box_half_extents,
                box_valid,
            ],
            outputs=[aabb_lower, aabb_upper],
        )

        shape_world_np = np.zeros(n_edges, dtype=np.int32)
        shape_collision_group_np = np.ones(n_edges, dtype=np.int32)
        shape_flags_np = np.full(n_edges, int(ShapeFlags.COLLIDE_SHAPES), dtype=np.int32)
        # Drop COLLIDE_SHAPES on degenerate edges so SAP skips them entirely.
        valid_host = box_valid.numpy()
        shape_flags_np[valid_host == 0] = 0

        shape_world_wp = wp.array(shape_world_np, dtype=wp.int32)
        shape_collision_group_wp = wp.array(shape_collision_group_np, dtype=wp.int32)
        shape_flags_wp = wp.array(shape_flags_np, dtype=wp.int32)

        sap = BroadPhaseSAP(shape_world=shape_world_wp, shape_flags=shape_flags_wp)

        candidate_pair_count = wp.zeros(1, dtype=wp.int32)
        capacity = max(64, initial_pair_capacity_factor * n_edges)
        attempts = 0
        actual_pair_count = 0
        candidate_pair: wp.array | None = None
        while True:
            candidate_pair = wp.empty(capacity, dtype=wp.vec2i)
            sap.launch(
                shape_lower=aabb_lower,
                shape_upper=aabb_upper,
                shape_gap=None,
                shape_collision_group=shape_collision_group_wp,
                shape_world=shape_world_wp,
                shape_count=n_edges,
                candidate_pair=candidate_pair,
                candidate_pair_count=candidate_pair_count,
            )
            actual_pair_count = int(candidate_pair_count.numpy()[0])
            if actual_pair_count <= capacity:
                break
            attempts += 1
            if attempts > max_retries:
                # SAP truncates internally; we proceed with the truncated set
                # but warn via the result (caller can detect via
                # broad_phase_pair_count == capacity).
                actual_pair_count = capacity
                break
            capacity = max(actual_pair_count, capacity * 2)

        assert candidate_pair is not None  # for type-checkers

        eps = 1.0e-6 * max(diagonal, 1.0e-6)

        swallow_count_per_box = wp.zeros(n_edges, dtype=wp.int32)
        num_containers_per_edge = wp.zeros(n_edges, dtype=wp.int32)

        if actual_pair_count > 0:
            wp.launch(
                kernel=_count_swallowed_per_box_kernel,
                dim=actual_pair_count,
                inputs=[
                    candidate_pair,
                    candidate_pair_count,
                    vertices_wp,
                    edge_indices_wp,
                    box_center,
                    box_axis_dir,
                    box_axis_tang,
                    box_axis_normal,
                    box_half_extents,
                    box_valid,
                    eps,
                ],
                outputs=[swallow_count_per_box, num_containers_per_edge],
            )

        # Exclusive scan over swallow_count_per_box -> swallowed_offsets[1:]
        swallowed_offsets = wp.zeros(n_edges + 1, dtype=wp.int32)
        if n_edges > 0:
            # Inclusive scan on the n_edges values, written into offsets[1:].
            inclusive_view = swallowed_offsets[1:]
            wp.utils.array_scan(swallow_count_per_box, inclusive_view, inclusive=True)

        offsets_host = swallowed_offsets.numpy()
        total_pairs = int(offsets_host[-1])
        swallowed_indices = wp.zeros(max(total_pairs, 1), dtype=wp.int32)
        write_cursor = wp.zeros(n_edges, dtype=wp.int32)

        if actual_pair_count > 0 and total_pairs > 0:
            wp.launch(
                kernel=_scatter_swallowed_per_box_kernel,
                dim=actual_pair_count,
                inputs=[
                    candidate_pair,
                    candidate_pair_count,
                    vertices_wp,
                    edge_indices_wp,
                    box_center,
                    box_axis_dir,
                    box_axis_tang,
                    box_axis_normal,
                    box_half_extents,
                    box_valid,
                    swallowed_offsets,
                    eps,
                ],
                outputs=[write_cursor, swallowed_indices],
            )

        candidate_for_removal = wp.zeros(n_edges, dtype=wp.int32)
        wp.launch(
            kernel=_mark_candidates_kernel,
            dim=n_edges,
            inputs=[num_containers_per_edge],
            outputs=[candidate_for_removal],
        )

        candidate_host = candidate_for_removal.numpy().astype(bool)
        num_containers_host = num_containers_per_edge.numpy()
        swallow_count_host = swallow_count_per_box.numpy()
        swallowed_indices_host = swallowed_indices.numpy()[:total_pairs]

    return EdgeRedundancyResult(
        edge_indices=edge_indices_np.reshape(-1, 2),
        dihedral_angles=edge_angles_np,
        candidate_for_removal=candidate_host,
        num_containers_per_edge=num_containers_host,
        swallow_count_per_box=swallow_count_host,
        swallowed_offsets=offsets_host,
        swallowed_indices=swallowed_indices_host,
        broad_phase_pair_count=actual_pair_count,
        aabb_diagonal=diagonal,
        half_height=resolved_half_height,
        half_width=resolved_half_width,
    )


# -----------------------------------------------------------------------------
# Greedy CPU-side resolution of removal candidates
# -----------------------------------------------------------------------------


@dataclass
class EdgeResolutionResult:
    """Per-edge greedy decision returned by :func:`resolve_edge_removals`.

    Indices are aligned with :attr:`EdgeRedundancyResult.edge_indices`.

    Attributes:
        to_remove [-]: Boolean mask, ``True`` for edges scheduled for definitive
            removal. The "kept" edges are simply ``~to_remove``.
        kept [-]: Boolean mask, ``True`` for edges that were promoted to
            "definitely keep" during the greedy pass (containers that were
            actually used). Always disjoint from :attr:`to_remove`.
        order [-]: Sort order over boxes (descending by ``swallow_count_per_box``)
            used by the greedy loop; useful for debugging and reproducibility.
        angle_threshold_rad [rad]: The threshold that was applied per swallowed
            edge.
    """

    to_remove: np.ndarray
    kept: np.ndarray
    order: np.ndarray
    angle_threshold_rad: float


def resolve_edge_removals(
    result: EdgeRedundancyResult,
    *,
    angle_threshold_rad: float = math.radians(10.0),
) -> EdgeResolutionResult:
    """Greedy CPU resolution of edge-removal candidates.

    Walks boxes from highest to lowest ``swallow_count_per_box``. For each box:

    1. If the box's own container edge is already scheduled for removal by an
       earlier (larger) container, skip the box.
    2. Otherwise, promote the container edge to the "definitely keep" set so it
       cannot be removed by any later iteration.
    3. For each edge swallowed by this box, mark it for removal iff its
       dihedral angle is below ``angle_threshold_rad`` AND it is not already in
       the keep set. Edges with sharper dihedrals are left alone (they are
       likely structural, not redundant tessellation).

    Args:
        result: Output of :func:`find_redundant_edges`.
        angle_threshold_rad [rad]: Maximum dihedral angle for an edge to be
            considered for definitive removal. Defaults to 10 degrees.

    Returns:
        :class:`EdgeResolutionResult` with per-edge removal and keep masks.
    """
    n = len(result.edge_indices)
    to_remove = np.zeros(n, dtype=bool)
    kept = np.zeros(n, dtype=bool)
    if n == 0:
        return EdgeResolutionResult(
            to_remove=to_remove,
            kept=kept,
            order=np.zeros(0, dtype=np.int32),
            angle_threshold_rad=float(angle_threshold_rad),
        )

    swallow_count = result.swallow_count_per_box.astype(np.int64, copy=False)
    # Stable descending sort (negate to flip sense; np.argsort is stable on a
    # single key by default with kind="stable").
    order = np.argsort(-swallow_count, kind="stable").astype(np.int32, copy=False)

    offsets = result.swallowed_offsets
    indices = result.swallowed_indices
    angles = result.dihedral_angles
    threshold = float(angle_threshold_rad)

    for box_idx in order:
        # Boxes that swallow nothing cannot consolidate anything; due to the
        # descending sort, all remaining boxes also have count 0.
        if swallow_count[box_idx] == 0:
            break
        if to_remove[box_idx]:
            # Container edge is already scheduled for removal -> skip.
            continue

        kept[box_idx] = True

        lo = int(offsets[box_idx])
        hi = int(offsets[box_idx + 1])
        if hi <= lo:
            continue
        swallowed = indices[lo:hi]
        # Apply both gating rules in one vectorized pass over this box's slice:
        #   - angle below threshold,
        #   - not already promoted to "definitely keep".
        flag = (angles[swallowed] < threshold) & (~kept[swallowed])
        if np.any(flag):
            to_remove[swallowed[flag]] = True

    # Invariant: an edge cannot be in both sets.
    assert not np.any(kept & to_remove), "kept and to_remove overlap"

    return EdgeResolutionResult(
        to_remove=to_remove,
        kept=kept,
        order=order,
        angle_threshold_rad=float(angle_threshold_rad),
    )


__all__ = [
    "EdgeRedundancyResult",
    "EdgeResolutionResult",
    "find_redundant_edges",
    "resolve_edge_removals",
]
