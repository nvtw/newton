# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Triangle-mesh queries on an :class:`~newton._src.geometry.implicit_aabb_tree.ImplicitAABBTree`.

:class:`ImplicitTriangleTree` builds the tree over the triangles of an indexed
mesh and provides fused closest-hit ray casts and closest-point queries, the
counterparts of ``wp.mesh_query_ray`` and ``wp.mesh_query_point``::

    tri_tree = ImplicitTriangleTree(points, indices)

    @wp.kernel
    def cast(mesh: ImplicitTriangleTreeData, ...):
        face, t, u, v = implicit_triangle_tree_query_ray(mesh, origin, direction, 1.0e30)

After moving ``points`` call :meth:`ImplicitTriangleTree.refit` (small motion)
or :meth:`ImplicitTriangleTree.build` (large motion or topology-preserving
remeshing); both are CUDA-graph capturable.
"""

from __future__ import annotations

import warp as wp

from .implicit_aabb_tree import (
    ImplicitAABBTree,
    ImplicitAABBTreeData,
    create_implicit_tree_point_closest,
    create_implicit_tree_ray_closest,
)

wp.set_module_options({"enable_backward": False})


@wp.struct
class ImplicitTriangleTreeData:
    """Kernel-side view of an :class:`ImplicitTriangleTree`."""

    tree: ImplicitAABBTreeData
    points: wp.array[wp.vec3]
    indices: wp.array[wp.int32]


@wp.kernel(enable_backward=False)
def _triangle_bounds(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
):
    f = wp.tid()
    a = points[indices[3 * f]]
    b = points[indices[3 * f + 1]]
    c = points[indices[3 * f + 2]]
    lowers[f] = wp.min(wp.min(a, b), c)
    uppers[f] = wp.max(wp.max(a, b), c)


@wp.func
def _ray_triangle(
    origin: wp.vec3, direction: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, max_t: wp.float32
) -> wp.vec3:
    """Moeller-Trumbore intersection: ``(t, u, v)`` with ``t < 0`` on a miss."""
    miss = wp.vec3(-1.0, 0.0, 0.0)
    e1 = b - a
    e2 = c - a
    p = wp.cross(direction, e2)
    det = wp.dot(e1, p)
    if wp.abs(det) < 1.0e-20:
        return miss
    inv = 1.0 / det
    s = origin - a
    u = wp.dot(s, p) * inv
    if u < 0.0 or u > 1.0:
        return miss
    q = wp.cross(s, e1)
    v = wp.dot(direction, q) * inv
    if v < 0.0 or u + v > 1.0:
        return miss
    t = wp.dot(e2, q) * inv
    if t < 0.0 or t >= max_t:
        return miss
    return wp.vec3(t, u, v)


@wp.func
def _closest_point_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
    """Barycentric weights ``(u, v, w)`` of the point on triangle ``abc`` closest to ``p``
    (the point is ``u a + v b + w c``; Ericson, Real-Time Collision Detection 5.1.5)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def _face_vertices(mesh: ImplicitTriangleTreeData, face: wp.int32) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    return (
        mesh.points[mesh.indices[3 * face]],
        mesh.points[mesh.indices[3 * face + 1]],
        mesh.points[mesh.indices[3 * face + 2]],
    )


@wp.func
def _face_ray_hit(
    face: wp.int32, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32, mesh: ImplicitTriangleTreeData
) -> wp.float32:
    a, b, c = _face_vertices(mesh, face)
    return _ray_triangle(origin, direction, a, b, c, max_t)[0]


@wp.func
def _face_distance(face: wp.int32, p: wp.vec3, max_dist: wp.float32, mesh: ImplicitTriangleTreeData) -> wp.float32:
    a, b, c = _face_vertices(mesh, face)
    w = _closest_point_triangle(p, a, b, c)
    return wp.length(p - (w[0] * a + w[1] * b + w[2] * c))


_mesh_ray_closest = create_implicit_tree_ray_closest(_face_ray_hit)
_mesh_point_closest = create_implicit_tree_point_closest(_face_distance)


@wp.func
def implicit_triangle_tree_query_ray(
    mesh: ImplicitTriangleTreeData, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> tuple[wp.int32, wp.float32, wp.float32, wp.float32]:
    """Closest ray hit: ``(face, t, u, v)``, ``face == -1`` on a miss.

    The hit point is ``origin + t * direction``, or equivalently
    ``u a + v b + (1 - u - v) c`` for the face's vertices ``a, b, c`` (the
    barycentric convention of ``wp.mesh_query_ray``).
    """
    face, t = _mesh_ray_closest(mesh.tree, origin, direction, max_t, mesh)
    u = wp.float32(0.0)
    v = wp.float32(0.0)
    if face >= wp.int32(0):
        a, b, c = _face_vertices(mesh, face)
        hit = _ray_triangle(origin, direction, a, b, c, 1.0e30)
        u = 1.0 - hit[1] - hit[2]
        v = hit[1]
    return face, t, u, v


@wp.func
def implicit_triangle_tree_query_point(
    mesh: ImplicitTriangleTreeData, point: wp.vec3, max_dist: wp.float32
) -> tuple[wp.int32, wp.float32, wp.vec3]:
    """Closest point on the mesh within ``max_dist``: ``(face, distance, closest_point)``.

    ``face == -1`` when no triangle lies within ``max_dist``.
    """
    face, dist = _mesh_point_closest(mesh.tree, point, max_dist, mesh)
    closest = point
    if face >= wp.int32(0):
        a, b, c = _face_vertices(mesh, face)
        w = _closest_point_triangle(point, a, b, c)
        closest = w[0] * a + w[1] * b + w[2] * c
    return face, dist, closest


class ImplicitTriangleTree:
    """An :class:`ImplicitAABBTree` over the triangles of an indexed mesh.

    The tree is built on construction. Pass :attr:`data` to kernels that call
    :func:`implicit_triangle_tree_query_ray` or
    :func:`implicit_triangle_tree_query_point`, or ``data.tree`` to the generic tree
    queries (items are face indices).

    Args:
        points: Vertex positions; the tree keeps a reference, so moved points
            only need :meth:`refit` or :meth:`build`.
        indices: Flat triangle vertex indices, three per face.
        device: Device; defaults to that of ``points``.
    """

    def __init__(self, points: wp.array[wp.vec3], indices: wp.array[wp.int32], device: wp.DeviceLike = None) -> None:
        if indices.shape[0] % 3 != 0:
            raise ValueError("indices must hold three vertex indices per triangle")
        self.device = wp.get_device(device if device is not None else points.device)
        self.points = points
        self.indices = indices
        self.face_count = indices.shape[0] // 3
        self.lowers = wp.empty(max(self.face_count, 1), dtype=wp.vec3, device=self.device)
        self.uppers = wp.empty(max(self.face_count, 1), dtype=wp.vec3, device=self.device)
        self.tree = ImplicitAABBTree(max(self.face_count, 1), device=self.device)
        self.data = ImplicitTriangleTreeData()
        """Kernel-side view to pass to kernels using the ``implicit_triangle_tree_query_*`` functions."""
        self.data.tree = self.tree.data
        self.data.points = points
        self.data.indices = indices
        self.build()

    def _update_bounds(self) -> None:
        if self.face_count > 0:
            wp.launch(
                _triangle_bounds,
                dim=self.face_count,
                inputs=[self.points, self.indices, self.lowers, self.uppers],
                device=self.device,
                record_tape=False,
            )

    def build(self) -> None:
        """Rebuild the tree from the current vertex positions."""
        self._update_bounds()
        self.tree.build(self.lowers, self.uppers, count=self.face_count)
        self.data.tree = self.tree.data

    def refit(self) -> None:
        """Update node bounds after the vertices moved, keeping the tree layout."""
        self._update_bounds()
        self.tree.refit(self.lowers, self.uppers)
