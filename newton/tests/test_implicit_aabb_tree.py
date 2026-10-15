# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit AABB tree queries against brute force, including refit and graph replay."""

import unittest

import numpy as np
import warp as wp

from newton._src.geometry.implicit_aabb_tree import (
    ImplicitAABBTree,
    ImplicitAABBTreeData,
    implicit_tree_query_aabb,
    implicit_tree_query_aabb_next,
    implicit_tree_query_point,
    implicit_tree_query_point_next,
    implicit_tree_query_ray,
    implicit_tree_query_ray_next,
    implicit_tree_query_self,
    implicit_tree_query_self_next,
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


@wp.kernel
def _closest_box_along_ray(
    tree: ImplicitAABBTreeData,
    ro: wp.array[wp.vec3],
    rd: wp.array[wp.vec3],
    out_t: wp.array[wp.float32],
    out_item: wp.array[wp.int32],
):
    i = wp.tid()
    best = int(-1)
    q = implicit_tree_query_ray(tree, ro[i], rd[i], 1.0e30)
    while q.item >= 0:
        # Every reported box is entered before max_t; shrinking prunes the rest.
        q.max_t = q.t
        best = q.item
        q = implicit_tree_query_ray_next(tree, q)
    out_t[i] = q.max_t
    out_item[i] = best


@wp.kernel
def _nearest_box(
    tree: ImplicitAABBTreeData,
    lo: wp.array[wp.vec3],
    hi: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    out_dist: wp.array[wp.float32],
):
    i = wp.tid()
    p = points[i]
    q = implicit_tree_query_point(tree, p, 1.0e30)
    while q.item >= 0:
        d = wp.max(wp.max(lo[q.item] - p, p - hi[q.item]), wp.vec3(0.0))
        q.max_dist = wp.min(q.max_dist, wp.length(d))
        q = implicit_tree_query_point_next(tree, q)
    out_dist[i] = q.max_dist


@wp.kernel
def _self_pairs(tree: ImplicitAABBTreeData, pairs: wp.array[wp.vec2i], count: wp.array[wp.int32]):
    slot = wp.tid()
    a = tree.leaf_item[slot]
    q = implicit_tree_query_self(tree, slot)
    while q.item >= 0:
        k = wp.atomic_add(count, 0, 1)
        if k < pairs.shape[0]:
            pairs[k] = wp.vec2i(wp.min(a, q.item), wp.max(a, q.item))
        q = implicit_tree_query_self_next(tree, q)


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

        # Closest box hit along random rays.
        ro_np = rng.uniform(-4.0, 4.0, (nq, 3)).astype(np.float32)
        rd_np = rng.standard_normal((nq, 3)).astype(np.float32)
        rd_np /= np.linalg.norm(rd_np, axis=1, keepdims=True)
        out_t = wp.zeros(nq, dtype=wp.float32, device=device)
        out_item = wp.zeros(nq, dtype=wp.int32, device=device)
        wp.launch(
            _closest_box_along_ray,
            dim=nq,
            inputs=[
                tree.data,
                wp.array(ro_np, dtype=wp.vec3, device=device),
                wp.array(rd_np, dtype=wp.vec3, device=device),
            ],
            outputs=[out_t, out_item],
            device=device,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / rd_np[:, None]
            t1 = (lo_np[None, :n] - ro_np[:, None]) * inv
            t2 = (hi_np[None, :n] - ro_np[:, None]) * inv
        t_in = np.maximum(np.minimum(t1, t2).max(-1), 0.0)
        t_out = np.maximum(t1, t2).min(-1)
        t_ref = np.where(t_in <= t_out, t_in, 1.0e30).min(1, initial=1.0e30)
        np.testing.assert_allclose(out_t.numpy(), t_ref, rtol=1e-5, atol=1e-5)

        # Nearest box distance from random points.
        p_np = rng.uniform(-6.0, 6.0, (nq, 3)).astype(np.float32)
        out_d = wp.zeros(nq, dtype=wp.float32, device=device)
        wp.launch(
            _nearest_box,
            dim=nq,
            inputs=[tree.data, lo, hi, wp.array(p_np, dtype=wp.vec3, device=device)],
            outputs=[out_d],
            device=device,
        )
        gap = np.maximum(np.maximum(lo_np[None, :n] - p_np[:, None], p_np[:, None] - hi_np[None, :n]), 0.0)
        np.testing.assert_allclose(
            out_d.numpy(), np.linalg.norm(gap, axis=-1).min(1, initial=1.0e30), rtol=1e-5, atol=1e-5
        )

        # Self overlap reports every overlapping pair exactly once.
        pairs = wp.empty(n * n, dtype=wp.vec2i, device=device)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_self_pairs, dim=tree.leaf_count, inputs=[tree.data, pairs, count], device=device)
        k = int(count.numpy()[0])
        got = [tuple(p) for p in pairs.numpy()[:k]]
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
        np.testing.assert_array_equal(tree.leaf_item.numpy()[:n], np.arange(n))
        self._check_all(tree, lo2_np, hi2_np, n, rng)

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
