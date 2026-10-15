# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit triangle tree ray and closest-point queries against wp.Mesh."""

import unittest

import numpy as np
import warp as wp

from newton.geometry import (
    ImplicitBvhMesh,
    ImplicitBvhMeshData,
    implicit_bvh_mesh_query_point,
    implicit_bvh_mesh_query_ray,
)


@wp.kernel
def _rays(
    mesh: ImplicitBvhMeshData,
    ref: wp.uint64,
    ro: wp.array[wp.vec3],
    rd: wp.array[wp.vec3],
    out: wp.array[wp.vec4],
    out_ref: wp.array[wp.vec4],
):
    i = wp.tid()
    face, t, u, v = implicit_bvh_mesh_query_ray(mesh, ro[i], rd[i], 1.0e30)
    out[i] = wp.vec4(float(face), t, u, v)
    q = wp.mesh_query_ray(ref, ro[i], rd[i], 1.0e30)
    out_ref[i] = wp.vec4(wp.where(q.result, float(q.face), -1.0), wp.where(q.result, q.t, 1.0e30), q.u, q.v)


@wp.kernel
def _points(
    mesh: ImplicitBvhMeshData,
    ref: wp.uint64,
    ps: wp.array[wp.vec3],
    out: wp.array[wp.vec2],
    out_ref: wp.array[wp.float32],
):
    i = wp.tid()
    _face, dist, closest = implicit_bvh_mesh_query_point(mesh, ps[i], 1.0e30)
    out[i] = wp.vec2(dist, wp.length(closest - ps[i]))
    q = wp.mesh_query_point_no_sign(ref, ps[i], 1.0e30)
    out_ref[i] = wp.length(wp.mesh_eval_position(ref, q.face, q.u, q.v) - ps[i])


def _terrain(rng, r=40):
    x, y = np.meshgrid(np.arange(r, dtype=np.float32), np.arange(r, dtype=np.float32), indexing="ij")
    z = np.sin(0.3 * x) * np.cos(0.2 * y) + 0.2 * rng.standard_normal(x.shape)
    pts = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(r * r).reshape(r, r)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return pts, tris[rng.permutation(len(tris))].astype(np.int32).ravel(), r


class TestImplicitBvhMesh(unittest.TestCase):
    def test_matches_warp_mesh(self):
        if not wp.is_cuda_available():
            self.skipTest("requires CUDA")
        device = "cuda:0"
        rng = np.random.default_rng(2)
        pts_np, idx_np, r = _terrain(rng)
        points = wp.array(pts_np, dtype=wp.vec3, device=device)
        indices = wp.array(idx_np, dtype=wp.int32, device=device)
        tri = ImplicitBvhMesh(points, indices)
        nq = 2000
        for _deformed in range(2):
            ref = wp.Mesh(points=points, indices=indices)
            ro = np.column_stack([rng.uniform(0, r, nq), rng.uniform(0, r, nq), np.full(nq, 5.0)]).astype(np.float32)
            rd = np.column_stack([rng.normal(0, 0.4, nq), rng.normal(0, 0.4, nq), -np.ones(nq)]).astype(np.float32)
            rd /= np.linalg.norm(rd, axis=1, keepdims=True)
            out = wp.zeros(nq, dtype=wp.vec4, device=device)
            out_ref = wp.zeros(nq, dtype=wp.vec4, device=device)
            wp.launch(
                _rays,
                dim=nq,
                inputs=[
                    tri.data,
                    ref.id,
                    wp.array(ro, dtype=wp.vec3, device=device),
                    wp.array(rd, dtype=wp.vec3, device=device),
                ],
                outputs=[out, out_ref],
                device=device,
            )
            got, want = out.numpy(), out_ref.numpy()
            np.testing.assert_allclose(got[:, 1], want[:, 1], rtol=1e-4, atol=1e-4)
            same_face = got[:, 0] == want[:, 0]
            self.assertGreater(same_face.mean(), 0.99)
            np.testing.assert_allclose(got[same_face, 2:], want[same_face, 2:], atol=1e-3)

            ps = rng.uniform([-3, -3, -3], [r + 3, r + 3, 3], (nq, 3)).astype(np.float32)
            d = wp.zeros(nq, dtype=wp.vec2, device=device)
            d_ref = wp.zeros(nq, dtype=wp.float32, device=device)
            wp.launch(
                _points,
                dim=nq,
                inputs=[tri.data, ref.id, wp.array(ps, dtype=wp.vec3, device=device)],
                outputs=[d, d_ref],
                device=device,
            )
            # The reported distance is that of the reported closest point, and both match wp.Mesh.
            np.testing.assert_allclose(d.numpy()[:, 0], d.numpy()[:, 1], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(d.numpy()[:, 0], d_ref.numpy(), rtol=1e-4, atol=1e-4)

            # Deform and refit; queries must follow the moved vertices.
            pts_np = pts_np + np.column_stack(
                [np.zeros((len(pts_np), 2)), 0.3 * rng.standard_normal(len(pts_np))]
            ).astype(np.float32)
            points.assign(pts_np)
            tri.refit()


if __name__ == "__main__":
    unittest.main()
