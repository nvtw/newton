# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverPhoenX integration test for the first-class TRIANGLE primitive.

PhoenX delegates rigid-body collision detection to the shared
``newton.CollisionPipeline`` / ``NarrowPhase``, so adding
:data:`newton.GeoType.TRIANGLE` as a routable primitive in the support
function is enough to make it work end-to-end with the PhoenX solver.

This test confirms that:

1. PhoenX accepts a model containing ``GeoType.TRIANGLE`` shapes without
   raising during construction or stepping.
2. A sphere dropped onto a horizontally-placed triangle comes to rest on
   the triangle face under gravity (penetration is bounded and vertical
   speed decays).
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton

GRAVITY = 9.81


def _make_sphere_on_triangle_model() -> newton.Model:
    """Static triangle (face up) + dynamic sphere just above the face."""
    mb = newton.ModelBuilder()

    # Right triangle in the world XY plane with vertex A at the origin.
    # ``add_shape_triangle`` derives the canonical local frame from the
    # three points and orients the triangle so its face normal points
    # along (B - A) x (C - A) -- here world +Z.
    mb.add_shape_triangle(
        body=-1,
        point_a=wp.vec3(0.0, 0.0, 0.0),
        point_b=wp.vec3(2.0, 0.0, 0.0),
        point_c=wp.vec3(0.0, 2.0, 0.0),
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5),
    )

    body = mb.add_body(
        xform=wp.transform(p=wp.vec3(0.5, 0.5, 0.3), q=wp.quat_identity()),
    )
    mb.add_shape_sphere(
        body,
        radius=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5),
    )

    mb.gravity = -GRAVITY
    return mb.finalize()


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "SolverPhoenX tests run on CUDA only.",
)
class TestTrianglePrimitivePhoenX(unittest.TestCase):
    """SolverPhoenX must accept and simulate scenes with TRIANGLE shapes."""

    def test_sphere_settles_on_triangle(self) -> None:
        model = _make_sphere_on_triangle_model()
        solver = newton.solvers.SolverPhoenX(model, substeps=8, solver_iterations=16)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        contacts = model.contacts(collision_pipeline=collision_pipeline)

        dt = 1.0 / 60.0
        for _ in range(120):  # 2 s
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        body_qd = state_0.body_qd.numpy()
        sphere_z = float(body_q[0, 2])
        speed = float(np.linalg.norm(body_qd[0, :3]))

        # Sphere of radius 0.1 resting on a triangle face at z=0 should sit
        # with its center near z=0.1. Triangle is a thin double-sided shell so
        # we accept a generous tolerance.
        self.assertGreater(
            sphere_z,
            -0.05,
            msg=f"Sphere passed through triangle: z={sphere_z:.3f}",
        )
        self.assertLess(
            sphere_z,
            0.2,
            msg=f"Sphere did not settle near triangle face: z={sphere_z:.3f}",
        )
        self.assertLess(
            speed,
            0.5,
            msg=f"Sphere still moving after settling: |v|={speed:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
