# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit AABB tree vs ``wp.Bvh`` vs ``wp.Mesh`` BVH: build, refit and queries.

Run::

    uv run python local_studies/implicit_tree_bvh/bench_bvh.py [--sizes 1000 10000 100000 1000000]

GPU timings are CUDA-graph replays (best of several trials of many replays);
operations that cannot be captured are timed eagerly (best of a few calls,
synchronized). Each timing first waits until no other process uses the GPU.
Results are printed and written to ``results.json`` next to this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(__file__))

from gpu_idle import wait_gpu_idle

import newton._src.geometry.implicit_aabb_tree as it
from newton._src.geometry.implicit_aabb_tree import (
    ImplicitAABBTree,
    ImplicitAABBTreeData,
    _aabb_disjoint,
    implicit_tree_next_subtree,
    implicit_tree_query_aabb,
    implicit_tree_query_aabb_next,
    implicit_tree_query_ray,
    implicit_tree_query_ray_next,
    implicit_tree_query_self,
    implicit_tree_query_self_next,
)

wp.config.quiet = True
DEVICE = "cuda:0"
INF = float(np.finfo(np.float32).max)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def k_tri_bounds(points: wp.array[wp.vec3], tris: wp.array[wp.int32], lo: wp.array[wp.vec3], hi: wp.array[wp.vec3]):
    t = wp.tid()
    a = points[tris[3 * t]]
    b = points[tris[3 * t + 1]]
    c = points[tris[3 * t + 2]]
    lo[t] = wp.min(wp.min(a, b), c)
    hi[t] = wp.max(wp.max(a, b), c)


@wp.kernel
def k_jitter_boxes(
    lo0: wp.array[wp.vec3], hi0: wp.array[wp.vec3], phase: float, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3]
):
    i = wp.tid()
    d = wp.vec3(wp.sin(float(i) * 0.37 + phase), wp.cos(float(i) * 0.11 + phase), wp.sin(float(i) * 0.05)) * 0.05
    lo[i] = lo0[i] + d
    hi[i] = hi0[i] + d


# --- AABB overlap: count hits per query box -------------------------------


@wp.kernel
def k_aabb_implicit(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    q = implicit_tree_query_aabb(tree, qlo[i], qhi[i])
    while q.item >= 0:
        n += 1
        q = implicit_tree_query_aabb_next(tree, q)
    counts[i] = n


@wp.kernel
def k_aabb_implicit_inline(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    """Hand-fused traversal (hit handling inside the loop), the upper bound for the iterator API."""
    i = wp.tid()
    lo = qlo[i]
    hi = qhi[i]
    n = int(0)
    node = wp.int32(1)
    while node != 0:
        r = tree.nodes[node]
        if _aabb_disjoint(r.lower, r.upper, lo, hi):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * 2
        else:
            n += 1  # item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_tree_next_subtree(node)
    counts[i] = n


@wp.kernel
def k_aabb_bvh(bvh: wp.uint64, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]):
    i = wp.tid()
    n = int(0)
    item = int(0)
    q = wp.bvh_query_aabb(bvh, qlo[i], qhi[i])
    while wp.bvh_query_next(q, item):
        n += 1
    counts[i] = n


@wp.kernel
def k_aabb_mesh(mesh: wp.uint64, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]):
    i = wp.tid()
    n = int(0)
    face = int(0)
    q = wp.mesh_query_aabb(mesh, qlo[i], qhi[i])
    while wp.mesh_query_aabb_next(q, face):
        n += 1
    counts[i] = n


# --- Rays against boxes ------------------------------------------------------


@wp.func
def ray_box_t(o: wp.vec3, inv_d: wp.vec3, lo: wp.vec3, hi: wp.vec3, max_t: float) -> float:
    t1 = wp.cw_mul(lo - o, inv_d)
    t2 = wp.cw_mul(hi - o, inv_d)
    tn = wp.min(t1, t2)
    tf = wp.max(t1, t2)
    t_in = wp.max(wp.max(tn[0], tn[1]), wp.max(tn[2], 0.0))
    t_out = wp.min(wp.min(tf[0], tf[1]), wp.min(tf[2], max_t))
    if t_in <= t_out:
        return t_in
    return -1.0


@wp.kernel
def k_ray_all_implicit(
    tree: ImplicitAABBTreeData,
    ro: wp.array[wp.vec3],
    rd: wp.array[wp.vec3],
    max_t: float,
    counts: wp.array[wp.int32],
):
    i = wp.tid()
    n = int(0)
    q = implicit_tree_query_ray(tree, ro[i], rd[i], max_t)
    while q.item >= 0:
        n += 1
        q = implicit_tree_query_ray_next(tree, q)
    counts[i] = n


@wp.kernel
def k_ray_all_bvh(
    bvh: wp.uint64, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], max_t: float, counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    item = int(0)
    q = wp.bvh_query_ray(bvh, ro[i], rd[i])
    while wp.bvh_query_next(q, item, max_t):
        n += 1
    counts[i] = n


@wp.kernel
def k_ray_closest_box_implicit(
    tree: ImplicitAABBTreeData, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], out_t: wp.array[float]
):
    i = wp.tid()
    q = implicit_tree_query_ray(tree, ro[i], rd[i], 1.0e30)
    while q.item >= 0:
        q.max_t = q.t  # every reported box is closer than max_t
        q = implicit_tree_query_ray_next(tree, q)
    out_t[i] = q.max_t


@wp.kernel
def k_ray_closest_box_bvh(
    bvh: wp.uint64,
    lo: wp.array[wp.vec3],
    hi: wp.array[wp.vec3],
    ro: wp.array[wp.vec3],
    rd: wp.array[wp.vec3],
    out_t: wp.array[float],
):
    i = wp.tid()
    o = ro[i]
    d = rd[i]
    inv_d = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    best = float(1.0e30)
    item = int(0)
    q = wp.bvh_query_ray(bvh, o, d)
    while wp.bvh_query_next(q, item, best):
        t = ray_box_t(o, inv_d, lo[item], hi[item], best)
        if t >= 0.0:
            best = t
    out_t[i] = best


# --- Rays against triangles (closest hit) -------------------------------------


@wp.func
def ray_tri_t(o: wp.vec3, d: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, max_t: float) -> float:
    e1 = b - a
    e2 = c - a
    p = wp.cross(d, e2)
    det = wp.dot(e1, p)
    if wp.abs(det) < 1.0e-12:
        return -1.0
    inv = 1.0 / det
    s = o - a
    u = wp.dot(s, p) * inv
    if u < 0.0 or u > 1.0:
        return -1.0
    qv = wp.cross(s, e1)
    v = wp.dot(d, qv) * inv
    if v < 0.0 or u + v > 1.0:
        return -1.0
    t = wp.dot(e2, qv) * inv
    if t < 0.0 or t >= max_t:
        return -1.0
    return t


@wp.kernel
def k_ray_tri_implicit(
    tree: ImplicitAABBTreeData,
    points: wp.array[wp.vec3],
    tris: wp.array[wp.int32],
    ro: wp.array[wp.vec3],
    rd: wp.array[wp.vec3],
    out_t: wp.array[float],
):
    i = wp.tid()
    o = ro[i]
    d = rd[i]
    q = implicit_tree_query_ray(tree, o, d, 1.0e30)
    while q.item >= 0:
        f = q.item
        t = ray_tri_t(o, d, points[tris[3 * f]], points[tris[3 * f + 1]], points[tris[3 * f + 2]], q.max_t)
        if t >= 0.0:
            q.max_t = t
        q = implicit_tree_query_ray_next(tree, q)
    out_t[i] = q.max_t


@wp.kernel
def k_ray_tri_bvh(
    bvh: wp.uint64,
    points: wp.array[wp.vec3],
    tris: wp.array[wp.int32],
    ro: wp.array[wp.vec3],
    rd: wp.array[wp.vec3],
    out_t: wp.array[float],
):
    i = wp.tid()
    o = ro[i]
    d = rd[i]
    best = float(1.0e30)
    f = int(0)
    q = wp.bvh_query_ray(bvh, o, d)
    while wp.bvh_query_next(q, f, best):
        t = ray_tri_t(o, d, points[tris[3 * f]], points[tris[3 * f + 1]], points[tris[3 * f + 2]], best)
        if t >= 0.0:
            best = t
    out_t[i] = best


@wp.kernel
def k_ray_tri_mesh(mesh: wp.uint64, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], out_t: wp.array[float]):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, ro[i], rd[i], 1.0e30)
    t = float(1.0e30)
    if q.result:
        t = q.t
    out_t[i] = t


# --- Self overlap: all overlapping pairs -------------------------------------


@wp.kernel
def k_self_implicit(tree: ImplicitAABBTreeData, counts: wp.array[wp.int32]):
    slot = wp.tid()
    n = int(0)
    q = implicit_tree_query_self(tree, slot)
    while q.item >= 0:
        n += 1
        q = implicit_tree_query_self_next(tree, q)
    counts[slot] = n


@wp.kernel
def k_self_implicit_inline(tree: ImplicitAABBTreeData, counts: wp.array[wp.int32]):
    """Hand-fused self traversal, same loop as BroadPhaseImplicitTree."""
    leaf = wp.tid()
    query_node = tree.leaf_base + leaf
    query = tree.nodes[query_node]
    lo = query.lower
    hi = query.upper
    span = query.max_span
    n = int(0)
    ancestor = query_node
    node = ancestor ^ 1
    stop = implicit_tree_next_subtree(node)
    while ancestor > 1:
        if node == stop:
            ancestor = ancestor >> 1
            node = ancestor ^ 1
            stop = implicit_tree_next_subtree(node)
            continue
        r = tree.nodes[node]
        if (
            r.max_span < span
            or (r.max_span == span and r.last_leaf <= leaf)
            or _aabb_disjoint(r.lower, r.upper, lo, hi)
        ):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * 2
        else:
            n += 1
            node = implicit_tree_next_subtree(node)
    counts[leaf] = n


@wp.kernel
def k_self_bvh(bvh: wp.uint64, lo: wp.array[wp.vec3], hi: wp.array[wp.vec3], counts: wp.array[wp.int32]):
    i = wp.tid()
    n = int(0)
    item = int(0)
    q = wp.bvh_query_aabb(bvh, lo[i], hi[i])
    while wp.bvh_query_next(q, item):
        if item > i:
            n += 1
    counts[i] = n


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def time_graph(fn, repeats=None, trials=5, budget_ms=150.0):
    """Best per-call ms of a captured graph; returns (ms, captured)."""
    try:
        with wp.ScopedCapture(device=DEVICE, force_module_load=False) as cap:
            fn()
        graph = cap.graph
    except Exception:
        wp.synchronize_device(DEVICE)
        return time_eager(fn), False
    wp.capture_launch(graph)
    wp.synchronize_device(DEVICE)
    if repeats is None:
        t0 = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize_device(DEVICE)
        one = max((time.perf_counter() - t0) * 1000.0, 1e-3)
        repeats = int(min(1000, max(3, budget_ms / one)))
    best = float("inf")
    for _ in range(trials):
        wait_gpu_idle()
        t0 = time.perf_counter()
        for _ in range(repeats):
            wp.capture_launch(graph)
        wp.synchronize_device(DEVICE)
        best = min(best, (time.perf_counter() - t0) * 1000.0 / repeats)
    return best, True


def time_eager(fn, trials=3):
    fn()
    wp.synchronize_device(DEVICE)
    best = float("inf")
    for _ in range(trials):
        wait_gpu_idle()
        t0 = time.perf_counter()
        fn()
        wp.synchronize_device(DEVICE)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def mempool_used():
    try:
        wp.synchronize_device(DEVICE)
        return wp.get_mempool_used_mem_current(DEVICE)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def scene_boxes(n, rng):
    side = n ** (1.0 / 3.0)
    c = rng.uniform(0.0, side, (n, 3)).astype(np.float32)
    h = rng.uniform(0.1, 0.5, (n, 3)).astype(np.float32)
    return c - h, c + h, side


def scene_terrain(n, rng):
    r = int(np.sqrt(n / 2)) + 1
    x, y = np.meshgrid(np.arange(r, dtype=np.float32), np.arange(r, dtype=np.float32), indexing="ij")
    z = 3.0 * np.sin(x * 0.05) * np.cos(y * 0.07) + 0.3 * rng.standard_normal(x.shape)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(r * r).reshape(r, r)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    # Shuffle triangle order so that the input order is not already coherent.
    tris = tris[rng.permutation(len(tris))]
    return pts, tris.astype(np.int32).ravel(), float(r)


def query_boxes_boxes(rng, q, side, half):
    c = rng.uniform(0.0, side, (q, 3)).astype(np.float32)
    return c - half, c + half


def query_boxes_terrain(rng, q, pts, half):
    c = pts[rng.integers(0, len(pts), q)] + rng.normal(0.0, 0.3, (q, 3)).astype(np.float32)
    return (c - half).astype(np.float32), (c + half).astype(np.float32)


def rays_boxes(rng, q, side):
    o = rng.uniform(0.0, side, (q, 3)).astype(np.float32)
    d = rng.standard_normal((q, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d


def rays_terrain(rng, q, side):
    o = np.column_stack([rng.uniform(0, side, q), rng.uniform(0, side, q), np.full(q, 20.0)]).astype(np.float32)
    d = np.column_stack([rng.normal(0, 0.5, q), rng.normal(0, 0.5, q), -np.ones(q)]).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def arr(x, dtype=wp.vec3):
    return wp.array(x, dtype=dtype, device=DEVICE)


def run_counts(kernel, n, inputs):
    counts = wp.zeros(n, dtype=wp.int32, device=DEVICE)

    def fn():
        wp.launch(kernel, dim=n, inputs=[*inputs, counts], device=DEVICE)

    fn()
    return counts, fn


def bench_case(kind, n, rng, num_queries, constructors, results):
    print(f"\n=== {kind} n={n} ===", flush=True)
    rec = {"scene": kind, "n": n, "build": {}, "refit": {}, "query": {}, "memory": {}, "capture": {}}
    points = tris = None
    if kind == "boxes":
        lo_np, hi_np, side = scene_boxes(n, rng)
        lo, hi = arr(lo_np), arr(hi_np)
    else:
        pts_np, tris_np, side = scene_terrain(n, rng)
        points, tris = arr(pts_np), arr(tris_np, wp.int32)
        n = len(tris_np) // 3
        rec["n"] = n
        lo = wp.empty(n, dtype=wp.vec3, device=DEVICE)
        hi = wp.empty(n, dtype=wp.vec3, device=DEVICE)
        wp.launch(k_tri_bounds, dim=n, inputs=[points, tris, lo, hi], device=DEVICE)
        lo_np, hi_np = lo.numpy(), hi.numpy()
        ms, _ = time_graph(lambda: wp.launch(k_tri_bounds, dim=n, inputs=[points, tris, lo, hi], device=DEVICE))
        rec["build"]["tri_bounds_kernel"] = ms

    # ---- implicit tree
    m0 = mempool_used()
    tree = ImplicitAABBTree(n, device=DEVICE)
    rec["memory"]["implicit"] = tree.memory_bytes
    tree.build(lo, hi)
    ms, cap = time_graph(lambda: tree.build(lo, hi))
    rec["build"]["implicit"] = ms
    rec["capture"]["implicit_build"] = cap
    ms, cap = time_graph(lambda: tree.refit(lo, hi))
    rec["refit"]["implicit"] = ms
    rec["capture"]["implicit_refit"] = cap
    # breakdown: internal-node reduction only
    ms, _ = time_graph(tree.update_internal_nodes)
    rec["refit"]["implicit_levels_only"] = ms
    del m0

    # ---- wp.Bvh variants
    bvhs = {}
    for c in constructors:
        try:
            m0 = mempool_used()
            bvh = wp.Bvh(lo, hi, constructor=c)
            m1 = mempool_used()
        except Exception as e:  # e.g. cubql not compiled in
            print(f"  wp.Bvh constructor={c} unavailable: {e}")
            continue
        if m0 is not None and m1 is not None and m1 > m0:
            rec["memory"][f"bvh_{c}"] = m1 - m0
        bvhs[c] = bvh
        trials = 3 if (c in ("sah", "median") and n >= 1_000_000) else 5
        rec["build"][f"bvh_{c}_ctor"] = time_eager(lambda c=c: wp.Bvh(lo, hi, constructor=c), trials=trials)
        ms, cap = time_graph(bvh.refit)
        rec["refit"][f"bvh_{c}"] = ms
        rec["capture"][f"bvh_{c}_refit"] = cap
    if "lbvh" in bvhs:
        b = bvhs["lbvh"]
        ms, cap = time_graph(lambda: b.rebuild("lbvh"))
        rec["build"]["bvh_lbvh_rebuild"] = ms
        rec["capture"]["bvh_lbvh_rebuild"] = cap
        # rebuild may have changed topology vs. construction; fine for queries.

    # ---- wp.Mesh
    mesh = None
    if kind == "terrain":
        m0 = mempool_used()
        mesh = wp.Mesh(points=points, indices=tris)
        m1 = mempool_used()
        if m0 is not None and m1 is not None and m1 > m0:
            rec["memory"]["mesh_lbvh"] = m1 - m0
        rec["build"]["mesh_lbvh_ctor"] = time_eager(lambda: wp.Mesh(points=points, indices=tris))
        if n <= 1_000_000:
            rec["build"]["mesh_sah_ctor"] = time_eager(
                lambda: wp.Mesh(points=points, indices=tris, bvh_constructor="sah"), trials=2
            )
        ms, cap = time_graph(mesh.refit)
        rec["refit"]["mesh_lbvh"] = ms
        rec["capture"]["mesh_refit"] = cap

    # ---- AABB queries
    for qname, half in (("aabb_small", 0.25), ("aabb_medium", 1.0)):
        if kind == "boxes":
            qlo_np, qhi_np = query_boxes_boxes(rng, num_queries, side, half)
        else:
            qlo_np, qhi_np = query_boxes_terrain(rng, num_queries, pts_np, half)
        qlo, qhi = arr(qlo_np), arr(qhi_np)
        res = {}
        ref = None
        variants = [
            ("implicit", k_aabb_implicit, [tree.data, qlo, qhi]),
            ("implicit_inline", k_aabb_implicit_inline, [tree.data, qlo, qhi]),
        ]
        variants += [(f"bvh_{c}", k_aabb_bvh, [b.id, qlo, qhi]) for c, b in bvhs.items()]
        if mesh is not None:
            variants.append(("mesh_lbvh", k_aabb_mesh, [mesh.id, qlo, qhi]))
        for vname, kern, inputs in variants:
            counts, fn = run_counts(kern, num_queries, inputs)
            c_np = counts.numpy()
            if ref is None:
                ref = c_np
                res["hits_per_query"] = float(c_np.mean())
            elif not np.array_equal(ref, c_np):
                raise RuntimeError(f"{kind} {n} {qname}: {vname} hit counts differ ({c_np.sum()} vs {ref.sum()})")
            ms, _ = time_graph(fn)
            res[vname] = ms
        if n <= 10_000:  # brute force reference on a subset
            sub = slice(0, 2000)
            ov = (
                (qlo_np[sub, None, :] <= hi_np[None, :, :]).all(-1)
                & (lo_np[None, :, :] <= qhi_np[sub, None, :]).all(-1)
            ).sum(1)
            if not np.array_equal(ov, ref[sub]):
                raise RuntimeError(f"{kind} {n} {qname}: brute force mismatch")
            res["brute_force_checked"] = True
        rec["query"][qname] = res
        print(f"  {qname}: {res}", flush=True)

    # ---- Ray queries
    if kind == "boxes":
        ro_np, rd_np = rays_boxes(rng, num_queries, side)
        ro, rd = arr(ro_np), arr(rd_np)
        for rname, max_t in (("ray_all_hits", 1.0e30), ("ray_short_all_hits", 2.0)):
            res = {}
            ref = None
            variants = [("implicit", k_ray_all_implicit, [tree.data, ro, rd, max_t])]
            variants += [(f"bvh_{c}", k_ray_all_bvh, [b.id, ro, rd, max_t]) for c, b in bvhs.items()]
            for vname, kern, inputs in variants:
                counts, fn = run_counts(kern, num_queries, inputs)
                c_np = counts.numpy()
                if ref is None:
                    ref = c_np
                    res["hits_per_query"] = float(c_np.mean())
                else:
                    mism = int((ref != c_np).sum())
                    res[f"{vname}_count_mismatch_rays"] = mism
                    if mism > num_queries * 1e-4:
                        raise RuntimeError(f"{rname} {vname}: {mism} rays differ")
                ms, _ = time_graph(fn)
                res[vname] = ms
            rec["query"][rname] = res
            print(f"  {rname}: {res}", flush=True)

        res = {}
        ref = None
        out_t = wp.zeros(num_queries, dtype=float, device=DEVICE)
        variants = [("implicit", k_ray_closest_box_implicit, [tree.data, ro, rd, out_t])]
        variants += [(f"bvh_{c}", k_ray_closest_box_bvh, [b.id, lo, hi, ro, rd, out_t]) for c, b in bvhs.items()]
        for vname, kern, inputs in variants:

            def fn(kern=kern, inputs=inputs):
                wp.launch(kern, dim=num_queries, inputs=inputs, device=DEVICE)

            fn()
            t_np = out_t.numpy().copy()
            if ref is None:
                ref = t_np
            elif not np.allclose(ref, t_np, rtol=1e-5, atol=1e-5):
                raise RuntimeError(f"ray_closest_box {vname}: results differ")
            ms, _ = time_graph(fn)
            res[vname] = ms
        rec["query"]["ray_closest_box"] = res
        print(f"  ray_closest_box: {res}", flush=True)
    else:
        ro_np, rd_np = rays_terrain(rng, num_queries, side)
        ro, rd = arr(ro_np), arr(rd_np)
        res = {}
        ref = None
        out_t = wp.zeros(num_queries, dtype=float, device=DEVICE)
        variants = [("implicit", k_ray_tri_implicit, [tree.data, points, tris, ro, rd, out_t])]
        variants += [(f"bvh_{c}", k_ray_tri_bvh, [b.id, points, tris, ro, rd, out_t]) for c, b in bvhs.items()]
        variants.append(("mesh_lbvh", k_ray_tri_mesh, [mesh.id, ro, rd, out_t]))
        for vname, kern, inputs in variants:

            def fn(kern=kern, inputs=inputs):
                wp.launch(kern, dim=num_queries, inputs=inputs, device=DEVICE)

            fn()
            t_np = out_t.numpy().copy()
            if ref is None:
                ref = t_np
                res["hit_fraction"] = float((t_np < 1e29).mean())
            else:
                bad = int((~np.isclose(ref, t_np, rtol=1e-4, atol=1e-4)).sum())
                res[f"{vname}_mismatch_rays"] = bad
                if bad > num_queries * 1e-3:
                    raise RuntimeError(f"ray_tri {vname}: {bad} rays differ")
            ms, _ = time_graph(fn)
            res[vname] = ms
        rec["query"]["ray_closest_tri"] = res
        print(f"  ray_closest_tri: {res}", flush=True)

    # ---- Self-overlap pairs
    res = {}
    variants = [
        ("implicit", k_self_implicit, [tree.data]),
        ("implicit_inline", k_self_implicit_inline, [tree.data]),
    ]
    if "lbvh" in bvhs:
        variants.append(("bvh_lbvh", k_self_bvh, [bvhs["lbvh"].id, lo, hi]))
    if "sah" in bvhs:
        variants.append(("bvh_sah", k_self_bvh, [bvhs["sah"].id, lo, hi]))
    ref = None
    for vname, kern, inputs in variants:
        dim = tree.leaf_count if vname.startswith("implicit") else n
        counts, fn = run_counts(kern, dim, inputs)
        total = int(counts.numpy().sum())
        if ref is None:
            ref = total
            res["pairs"] = total
        elif total != ref:
            raise RuntimeError(f"self pairs {vname}: {total} vs {ref}")
        ms, _ = time_graph(fn)
        res[vname] = ms
    rec["query"]["self_pairs"] = res
    print(f"  self_pairs: {res}", flush=True)
    print(f"  build: {rec['build']}", flush=True)
    print(f"  refit: {rec['refit']}", flush=True)
    print(f"  memory: {rec['memory']}", flush=True)
    print(f"  capture: {rec['capture']}", flush=True)
    results.append(rec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--scenes", nargs="+", default=["boxes", "terrain"])
    parser.add_argument("--queries", type=int, default=1 << 18)
    parser.add_argument("--constructors", nargs="+", default=["lbvh", "sah", "median", "cubql"])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = parser.parse_args()
    wp.init()
    it_mod = it  # ensure module is loaded before capture
    del it_mod
    results = []
    for kind in args.scenes:
        for n in args.sizes:
            bench_case(kind, n, np.random.default_rng(1234 + n), args.queries, args.constructors, results)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
    import gpu_idle  # noqa: PLC0415

    print(f"\nwaited for GPU idle: {gpu_idle.stats['waited_s']:.1f} s total")


if __name__ == "__main__":
    main()
