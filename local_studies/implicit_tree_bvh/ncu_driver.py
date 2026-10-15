# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Launch representative implicit-BVH and Warp BVH queries once each for Nsight Compute.

Every kernel is launched once untimed to compile and warm caches, then once more
between ``cuProfilerStart``/``cuProfilerStop``; run ncu with
``--profile-from-start off`` so only the second launches are captured. The
case names, in launch order (= report ID order), are written to the manifest
file given as the first argument.
"""

from __future__ import annotations

import ctypes
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.path.dirname(__file__))

import bench_bvh as b

from newton._src.geometry.implicit_bvh import ImplicitBvh
from newton._src.geometry.implicit_bvh_mesh import ImplicitBvhMesh

QUERIES = 1 << 18
_cuda = ctypes.CDLL("libcuda.so.1")


def terrain_cases(n, rng):
    pts_np, tris_np, side = b.scene_terrain(n, rng)
    points, tris = b.arr(pts_np), b.arr(tris_np, wp.int32)
    mesh_sah = wp.Mesh(points=points, indices=tris, bvh_constructor="sah")
    tri_tree = ImplicitBvhMesh(points, tris)
    ro_np, rd_np = b.rays_terrain(rng, QUERIES, side)
    ro, rd = b.arr(ro_np), b.arr(rd_np)
    ps = b.arr(
        np.column_stack(
            [rng.uniform(0, side, QUERIES), rng.uniform(0, side, QUERIES), rng.uniform(-6.0, 6.0, QUERIES)]
        ).astype(np.float32)
    )
    out = wp.zeros(QUERIES, dtype=float, device=b.DEVICE)
    keep = (points, tris, mesh_sah, tri_tree)
    return keep, [
        (f"terrain{n}_ray_implicit", b.k_ray_tri_implicit_mesh, [tri_tree.data, ro, rd, out]),
        (f"terrain{n}_ray_meshsah", b.k_ray_tri_mesh, [mesh_sah.id, ro, rd, out]),
        (f"terrain{n}_point_implicit", b.k_point_tri_implicit_mesh, [tri_tree.data, ps, out]),
        (f"terrain{n}_point_meshsah", b.k_point_tri_mesh, [mesh_sah.id, ps, out]),
    ]


def box_cases(n, rng):
    lo_np, hi_np, side = b.scene_boxes(n, rng)
    lo, hi = b.arr(lo_np), b.arr(hi_np)
    tree = ImplicitBvh(n, device=b.DEVICE)
    tree.build(lo, hi)
    bvh = wp.Bvh(lo, hi, constructor="sah")
    ro_np, rd_np = b.rays_boxes(rng, QUERIES, side)
    ro, rd = b.arr(ro_np), b.arr(rd_np)
    qlo_np, qhi_np = b.query_boxes_boxes(rng, QUERIES, side, 1.0)
    qlo, qhi = b.arr(qlo_np), b.arr(qhi_np)
    counts = wp.zeros(max(QUERIES, tree.slot_count), dtype=wp.int32, device=b.DEVICE)
    keep = (lo, hi, tree, bvh)
    return keep, [
        (f"boxes{n}_rayall_implicit", b.k_ray_all_implicit_fused, [tree.data, ro, rd, 1.0e30, counts]),
        (f"boxes{n}_rayall_bvhsah", b.k_ray_all_bvh, [bvh.id, ro, rd, 1.0e30, counts]),
        (f"boxes{n}_aabb_implicit", b.k_aabb_implicit, [tree.data, qlo, qhi, counts]),
        (f"boxes{n}_aabb_bvhsah", b.k_aabb_bvh, [bvh.id, qlo, qhi, counts]),
        (f"boxes{n}_self_implicit", b.k_self_implicit_fused, [tree.data, counts], tree.slot_count),
        (f"boxes{n}_self_bvhsah", b.k_self_bvh, [bvh.id, lo, hi, counts], n),
    ]


def main():
    wp.config.lineinfo = True  # source-level counters in the report
    wp.init()
    rng = np.random.default_rng(7)
    keep = []
    cases = []
    for make, n in ((terrain_cases, 1000), (terrain_cases, 100_000), (box_cases, 1000), (box_cases, 100_000)):
        k, c = make(n, rng)
        keep.append(k)
        cases += c
    launches = [(kernel, inputs, dim[0] if dim else QUERIES) for _name, kernel, inputs, *dim in cases]
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as stream:
            stream.write("\n".join(case[0] for case in cases) + "\n")
    for kernel, inputs, dim in launches:
        wp.launch(kernel, dim=dim, inputs=inputs, device=b.DEVICE)
    wp.synchronize_device(b.DEVICE)
    _cuda.cuProfilerStart()
    for kernel, inputs, dim in launches:
        wp.launch(kernel, dim=dim, inputs=inputs, device=b.DEVICE)
    wp.synchronize_device(b.DEVICE)
    _cuda.cuProfilerStop()
    print(f"profiled {len(launches)} launches")


if __name__ == "__main__":
    main()
