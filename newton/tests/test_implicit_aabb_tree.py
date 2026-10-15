# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit AABB tree queries against brute force, including refit and graph replay."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.implicit_aabb_tree import (
    ImplicitAABBTree,
    ImplicitAABBTreeData,
    create_implicit_tree_point_closest,
    create_implicit_tree_ray_closest,
    create_implicit_tree_ray_foreach,
    create_implicit_tree_self_foreach,
    implicit_tree_query_aabb,
    implicit_tree_query_aabb_next,
)

_MAX_HITS = 64


@wp.kernel
def _aabb_hits(
    tree: ImplicitAABBTreeData,
    qlo: wp.array[wp.vec3],
    qhi: wp.array[wp.vec3],
    hits: wp.array2d[wp.int32],
    counts: wp.array[wp.int32],
):
    i = wp.tid()
    n = int(0)
    q = implicit_tree_query_aabb(tree, qlo[i], qhi[i])
    while q.item >= 0:
        if n < _MAX_HITS:
            hits[i, n] = q.item
        n += 1
        q = implicit_tree_query_aabb_next(tree, q)
    counts[i] = n


@wp.struct
class _Boxes:
    lo: wp.array[wp.vec3]
    hi: wp.array[wp.vec3]


@wp.func
def _box_hit(item: int, o: wp.vec3, d: wp.vec3, max_t: float, boxes: _Boxes) -> float:
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    t1 = wp.cw_mul(boxes.lo[item] - o, inv)
    t2 = wp.cw_mul(boxes.hi[item] - o, inv)
    t_in = wp.max(wp.max(wp.min(t1[0], t2[0]), wp.min(t1[1], t2[1])), wp.max(wp.min(t1[2], t2[2]), 0.0))
    t_out = wp.min(wp.min(wp.max(t1[0], t2[0]), wp.max(t1[1], t2[1])), wp.min(wp.max(t1[2], t2[2]), max_t))
    if t_in <= t_out:
        return t_in
    return -1.0


@wp.func
def _box_distance(item: int, p: wp.vec3, max_dist: float, boxes: _Boxes) -> float:
    d = wp.max(wp.max(boxes.lo[item] - p, p - boxes.hi[item]), wp.vec3(0.0))
    return wp.length(d)


@wp.func
def _count_visit(item: int, t: float, state: int, context: int) -> int:
    return state + 1


@wp.struct
class _PairOut:
    pairs: wp.array[wp.vec2i]
    count: wp.array[wp.int32]


@wp.func
def _record_pair(item: int, other: int, state: int, out: _PairOut) -> int:
    k = wp.atomic_add(out.count, 0, 1)
    if k < out.pairs.shape[0]:
        out.pairs[k] = wp.vec2i(wp.min(item, other), wp.max(item, other))
    return state


_ray_closest = create_implicit_tree_ray_closest(_box_hit)
_ray_count = create_implicit_tree_ray_foreach(_count_visit)
_nearest = create_implicit_tree_point_closest(_box_distance)
_self_pairs = create_implicit_tree_self_foreach(_record_pair)


@wp.kernel
def _nearest_box(tree: ImplicitAABBTreeData, boxes: _Boxes, points: wp.array[wp.vec3], out_dist: wp.array[wp.float32]):
    i = wp.tid()
    _item, d = _nearest(tree, points[i], 1.0e30, boxes)
    out_dist[i] = d


@wp.kernel
def _all_pairs(tree: ImplicitAABBTreeData, out: _PairOut):
    _self_pairs(tree, wp.tid(), int(0), out)


@wp.kernel
def _fused_rays(
    tree: ImplicitAABBTreeData,
    boxes: _Boxes,
    ro: wp.array[wp.vec3],
    rd: wp.array[wp.vec3],
    out_t: wp.array[wp.float32],
    out_item: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
):
    i = wp.tid()
    item, t = _ray_closest(tree, ro[i], rd[i], 1.0e30, boxes)
    out_t[i] = t
    out_item[i] = item
    out_count[i] = _ray_count(tree, ro[i], rd[i], 1.0e30, int(0), int(0))


def _random_boxes(rng, n):
    c = rng.uniform(-3.0, 3.0, (n, 3)).astype(np.float32)
    h = rng.uniform(0.05, 0.6, (n, 3)).astype(np.float32)
    return c - h, c + h


class TestImplicitAABBTree(unittest.TestCase):
    def setUp(self):
        if not wp.is_cuda_available():
            self.skipTest("CUDA graph capture requires a CUDA device")
        self.device = wp.get_device("cuda:0")

    def _check_all(self, tree, lo_np, hi_np, n, rng):
        device = self.device
        lo = wp.array(lo_np, dtype=wp.vec3, device=device)
        hi = wp.array(hi_np, dtype=wp.vec3, device=device)
        nq = 200
        qlo_np, qhi_np = _random_boxes(rng, nq)
        qlo = wp.array(qlo_np, dtype=wp.vec3, device=device)
        qhi = wp.array(qhi_np, dtype=wp.vec3, device=device)
        hits = wp.full((nq, _MAX_HITS), -1, dtype=wp.int32, device=device)
        counts = wp.zeros(nq, dtype=wp.int32, device=device)
        wp.launch(_aabb_hits, dim=nq, inputs=[tree.data, qlo, qhi, hits, counts], device=device)
        overlap = (qlo_np[:, None] <= hi_np[None, :n]).all(-1) & (lo_np[None, :n] <= qhi_np[:, None]).all(-1)
        np.testing.assert_array_equal(counts.numpy(), overlap.sum(1))
        hits_np = hits.numpy()
        for i in range(nq):
            got = sorted(h for h in hits_np[i] if h >= 0)
            self.assertEqual(got, sorted(np.flatnonzero(overlap[i]).tolist()))

        # Closest box hit and every box entered along random rays.
        ro_np = rng.uniform(-4.0, 4.0, (nq, 3)).astype(np.float32)
        rd_np = rng.standard_normal((nq, 3)).astype(np.float32)
        rd_np /= np.linalg.norm(rd_np, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / rd_np[:, None]
            t1 = (lo_np[None, :n] - ro_np[:, None]) * inv
            t2 = (hi_np[None, :n] - ro_np[:, None]) * inv
        t_in = np.maximum(np.minimum(t1, t2).max(-1), 0.0)
        t_out = np.maximum(t1, t2).min(-1)
        t_ref = np.where(t_in <= t_out, t_in, 1.0e30).min(1, initial=1.0e30)
        boxes = _Boxes()
        boxes.lo = lo
        boxes.hi = hi
        out_t = wp.zeros(nq, dtype=wp.float32, device=device)
        out_item = wp.zeros(nq, dtype=wp.int32, device=device)
        out_count = wp.zeros(nq, dtype=wp.int32, device=device)
        wp.launch(
            _fused_rays,
            dim=nq,
            inputs=[
                tree.data,
                boxes,
                wp.array(ro_np, dtype=wp.vec3, device=device),
                wp.array(rd_np, dtype=wp.vec3, device=device),
            ],
            outputs=[out_t, out_item, out_count],
            device=device,
        )
        np.testing.assert_allclose(out_t.numpy(), t_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(out_count.numpy(), (t_in <= t_out).sum(1))

        # Nearest box distance from random points.
        p_np = rng.uniform(-6.0, 6.0, (nq, 3)).astype(np.float32)
        out_d = wp.zeros(nq, dtype=wp.float32, device=device)
        wp.launch(
            _nearest_box,
            dim=nq,
            inputs=[tree.data, boxes, wp.array(p_np, dtype=wp.vec3, device=device)],
            outputs=[out_d],
            device=device,
        )
        gap = np.maximum(np.maximum(lo_np[None, :n] - p_np[:, None], p_np[:, None] - hi_np[None, :n]), 0.0)
        np.testing.assert_allclose(
            out_d.numpy(), np.linalg.norm(gap, axis=-1).min(1, initial=1.0e30), rtol=1e-5, atol=1e-5
        )

        # Self overlap reports every overlapping pair exactly once.
        out = _PairOut()
        out.pairs = wp.empty(max(n * n, 1), dtype=wp.vec2i, device=device)
        out.count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_all_pairs, dim=tree.leaf_base, inputs=[tree.data, out], device=device)
        k = int(out.count.numpy()[0])
        got = [tuple(p) for p in out.pairs.numpy()[:k]]
        self.assertEqual(len(got), len(set(got)))
        ov = (lo_np[:n, None] <= hi_np[None, :n]).all(-1) & (lo_np[None, :n] <= hi_np[:n, None]).all(-1)
        a, b = np.nonzero(np.triu(ov, 1))
        self.assertEqual(set(got), set(zip(a.tolist(), b.tolist(), strict=True)))

    def test_queries_build_refit_and_graph_replay(self):
        device = self.device
        rng = np.random.default_rng(3)
        n = 300
        lo_np, hi_np = _random_boxes(rng, n)
        lo = wp.array(lo_np, dtype=wp.vec3, device=device)
        hi = wp.array(hi_np, dtype=wp.vec3, device=device)
        tree = ImplicitAABBTree(n + 37, device=device)
        tree.build(lo, hi)
        self._check_all(tree, lo_np, hi_np, n, rng)

        # A smaller item count reuses the same buffers.
        tree.build(lo, hi, count=n // 2)
        self._check_all(tree, lo_np, hi_np, n // 2, rng)

        # Captured build and refit replay against the current box contents.
        with wp.ScopedCapture(device=device) as build_capture:
            tree.build(lo, hi)
        with wp.ScopedCapture(device=device) as refit_capture:
            tree.refit(lo, hi)
        wp.capture_launch(build_capture.graph)
        self._check_all(tree, lo_np, hi_np, n, rng)
        lo2_np = lo_np + rng.uniform(-0.3, 0.3, (n, 3)).astype(np.float32)
        hi2_np = lo2_np + (hi_np - lo_np)
        lo.assign(lo2_np)
        hi.assign(hi2_np)
        wp.capture_launch(refit_capture.graph)
        self._check_all(tree, lo2_np, hi2_np, n, rng)
        wp.capture_launch(build_capture.graph)
        self._check_all(tree, lo2_np, hi2_np, n, rng)

        # Unsorted build keeps the input order and remains exact.
        tree.build(lo, hi, sort=False)
        # Left-to-right leaf positions map to heap slots across two depths.
        deep = 2 * tree.leaf_base - tree.deep_start
        positions = np.arange(tree.leaf_base)
        slots = (
            np.where(positions < deep, tree.deep_start + positions, tree.leaf_base + positions - deep) - tree.leaf_base
        )
        np.testing.assert_array_equal(tree.leaf_item.numpy()[slots[:n]], np.arange(n))
        self._check_all(tree, lo2_np, hi2_np, n, rng)

    def test_slack_unaligned_and_clustered(self):
        """Spare leaf slots, count splits and clustered boxes stay exact."""
        device = self.device
        rng = np.random.default_rng(11)
        n = 2000
        centers = rng.normal(0.0, 0.4, (n, 3)) + rng.choice([-2.5, 0.0, 2.5], size=(n, 3))
        half = rng.uniform(0.01, 0.2, (n, 3))
        lo_np = (centers - half).astype(np.float32)
        hi_np = (centers + half).astype(np.float32)
        lo = wp.array(lo_np, dtype=wp.vec3, device=device)
        hi = wp.array(hi_np, dtype=wp.vec3, device=device)
        for kwargs in ({}, {"slack": 2.0}, {"aligned": False}):
            with self.subTest(**kwargs):
                tree = ImplicitAABBTree(n, device=device, **kwargs)
                tree.build(lo, hi)
                self.assertEqual(int((tree.leaf_item.numpy() >= 0).sum()), n)
                self._check_all(tree, lo_np, hi_np, n, rng)
                for _ in range(3):
                    tree.refit(lo, hi)
                self._check_all(tree, lo_np, hi_np, n, rng)

    def test_tiny_and_empty_trees(self):
        device = self.device
        rng = np.random.default_rng(5)
        for n in (0, 1, 2, 3):
            lo_np, hi_np = _random_boxes(rng, max(n, 1))
            lo = wp.array(lo_np, dtype=wp.vec3, device=device)
            hi = wp.array(hi_np, dtype=wp.vec3, device=device)
            tree = ImplicitAABBTree(max(n, 1), device=device)
            tree.build(lo, hi, count=n)
            self._check_all(tree, lo_np, hi_np, n, rng)


if __name__ == "__main__":
    unittest.main()
