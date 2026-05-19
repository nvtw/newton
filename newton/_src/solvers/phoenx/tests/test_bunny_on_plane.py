# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vertex-level ground-plane penetration test for :class:`PhoenXWorld`.

Drops a single mesh bunny onto a ground plane and asserts that once the
scene has settled every vertex of the mesh lies at (or above) the
ground plane, within a tight sub-mm tolerance. This is the primary
regression for contact-count robustness: the bunny is a >1000-vertex
mesh so the ground-plane narrow phase emits a dense per-triangle
manifold (tens of points per frame), and the previous 6-slot-per-column
contact schema let large manifolds sink into the plane because it
split them across multiple colour classes that never saw each other's
impulses within a single PGS iteration. The per-pair contact column
schema loops every contact of the manifold serially (Gauss-Seidel
within the pair) so the normal impulse budget is correctly distributed
across every manifold point and the mesh stops at z = 0.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.tests.test_stacking import _PhoenXScene


def _icosahedron_vertices_and_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, indices)`` for a unit icosahedron.

    Dependency-free fallback when Newton's USD loader isn't installed.
    12 vertices + 20 faces -- enough contact points to exercise the
    dense-manifold pathway this test regresses.
    """
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    verts = np.array(
        [
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            0,
            11,
            5,
            0,
            5,
            1,
            0,
            1,
            7,
            0,
            7,
            10,
            0,
            10,
            11,
            1,
            5,
            9,
            5,
            11,
            4,
            11,
            10,
            2,
            10,
            7,
            6,
            7,
            1,
            8,
            3,
            9,
            4,
            3,
            4,
            2,
            3,
            2,
            6,
            3,
            6,
            8,
            3,
            8,
            9,
            4,
            9,
            5,
            2,
            4,
            11,
            6,
            2,
            10,
            8,
            6,
            7,
            9,
            8,
            1,
        ],
        dtype=np.int32,
    )
    return verts, faces


def _load_bunny_mesh() -> newton.Mesh:
    """Load the bunny triangle mesh, falling back to an icosahedron.

    The bunny has more vertices (dense manifold) so it's the stronger
    regression target; the icosahedron fallback still exercises the
    per-pair multi-contact column path.
    """
    try:
        from pxr import Usd

        import newton.usd as newton_usd  # noqa: PLC0415
    except ModuleNotFoundError:
        verts, faces = _icosahedron_vertices_and_indices()
        return newton.Mesh(verts, faces)
    stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
    return newton_usd.get_mesh(stage.GetPrimAtPath("/root/bunny"))


def _min_vertex_world_z(mesh: newton.Mesh, body_q: np.ndarray) -> float:
    """Return the minimum world-space z coordinate over every mesh vertex.

    ``body_q`` is Newton's 7-component pose ``(px, py, pz, qx, qy, qz,
    qw)`` -- the body-*origin* transform that the narrow phase and the
    viewer both use to place the mesh in world space. We deliberately
    read from ``state.body_q`` rather than PhoenX's COM-based
    ``bodies.position`` so the test sees exactly what a user watching
    the viewer would see; a mismatch between the two (caused by the
    contact kernels treating a body-origin anchor as COM-relative)
    would make the mesh visually dip into the plane even though the
    COM looks stable.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    position = np.asarray(body_q[:3], dtype=np.float64)
    qx, qy, qz, qw = (
        float(body_q[3]),
        float(body_q[4]),
        float(body_q[5]),
        float(body_q[6]),
    )
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    world = verts @ rot.T + position.reshape(1, 3)
    return float(world[:, 2].min())


@unittest.skipUnless(wp.is_cuda_available(), "PhoenX bunny-on-plane test requires CUDA")
class TestPhoenXBunnyOnPlane(unittest.TestCase):
    """No mesh vertex should penetrate the ground plane after settle.

    Bunny falls from z = 0.8 m, scene steps for 2.5 s at 120 Hz so the
    vertical transient decays fully; we then measure the minimum
    world-space z over every bunny vertex. The per-pair contact schema
    should keep that minimum at zero (sub-mm tolerance); a regression
    to the old multi-column split reliably sinks the bunny tens of mm
    below the plane because the graph colourer fragmented the mesh
    manifold across colour classes.
    """

    def test_no_vertex_penetrates_ground_plane(self) -> None:
        mesh = _load_bunny_mesh()

        scene = _PhoenXScene(
            fps=120,
            substeps=8,
            solver_iterations=16,
            friction=0.5,
        )
        scene.add_ground_plane()
        body = scene.mb.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.8), q=wp.quat_identity()),
        )
        scene.mb.add_shape_mesh(
            body,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
        )
        scene._newton_body_ids.append(body)
        scene.finalize()

        for _ in range(300):  # 2.5 s at 120 Hz
            scene.step()

        body_q = scene.state.body_q.numpy()[body]
        min_z = _min_vertex_world_z(mesh, body_q)

        self.assertTrue(
            np.isfinite(body_q).all(),
            f"bunny body_q non-finite after settle: body_q={body_q}",
        )
        # Tolerance: 0.5 mm. In practice the contact kernels -- once
        # the body-origin vs COM convention is handled correctly (see
        # :class:`BodyContainer.body_com` and
        # :func:`contact_prepare_for_iteration_at`) -- keep the
        # minimum vertex z pinned within a few micrometres of the
        # plane. A regression that reverts the origin/COM fix would
        # make the rendered mesh dip by ``|body_com|`` (~10 cm on the
        # bunny asset); a regression that reverts the per-pair
        # contact column split would fragment the dense manifold
        # across colour classes and sink the mesh by tens of mm. The
        # half-mm bound is a ~100x FP-noise buffer that trips on
        # either failure mode well before a user would see it in the
        # viewer.
        tolerance_m = 0.0005
        self.assertGreaterEqual(
            min_z,
            -tolerance_m,
            f"bunny pierces ground plane: min vertex z = {min_z:.4f} m (tolerance {tolerance_m:.4f} m)",
        )


if __name__ == "__main__":
    unittest.main()
