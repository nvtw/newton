# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-independent checks of penetration-free collision certification."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.penetration_free import (
    ConstraintFeature,
    halfspace_motion_fraction,
    screw_motion_delta,
)
from newton.geometry import PenetrationFreeConfig
from newton.solvers import SolverXPBD

_DT = 1.0 / 60.0


class TestPenetrationFreeContract(unittest.TestCase):
    def test_unsupported_solver_rejects_buffer(self):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(model, penetration_free=PenetrationFreeConfig())
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        pipeline.collide(state_0, contacts)
        solver = SolverXPBD(model)
        with self.assertRaisesRegex(ValueError, "penetration-free"):
            solver.step(state_0, state_1, None, contacts, _DT)

    def test_dynamic_infinite_plane_is_rejected(self):
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_plane(body=body, width=0.0, length=0.0)
        builder.add_shape_sphere(builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity())), radius=0.1)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "infinite plane"):
            newton.CollisionPipeline(model, penetration_free=PenetrationFreeConfig())

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            PenetrationFreeConfig(tolerance=-1.0).validate()
        with self.assertRaises(ValueError):
            PenetrationFreeConfig(min_motion_budget=0.5, max_motion_budget=0.1).validate()


@wp.kernel(enable_backward=False)
def _rotating_box_fraction_kernel(
    angle: float,
    limit: float,
    half: wp.vec3,
    lipschitz: float,
    out_fraction: wp.array[float],
):
    # A box with its center of mass on the plane normal axis rotates about
    # +y by ``angle``. Its lowest support dips below ``-limit`` mid-path and
    # returns above it at the end, so endpoint checks alone accept the motion.
    normal = wp.vec3(0.0, 0.0, -1.0)
    x0 = wp.vec3(0.0)
    q0 = wp.quat_identity()
    q1 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    dx, axis, theta = screw_motion_delta(x0, q0, x0, q1)
    feature = ConstraintFeature()
    feature.local_rot = wp.quat_identity()
    feature.shape_type = 7  # GeoType.BOX
    feature.scale = half
    feature.lipschitz = lipschitz
    out_fraction[0] = halfspace_motion_fraction(feature, normal, limit, x0, q0, dx, axis, theta)


class TestPenetrationFreeCertification(unittest.TestCase):
    def test_rotation_crossing_and_returning_is_truncated(self):
        half = np.array([0.5, 0.5, 0.1], dtype=np.float32)
        lipschitz = float(np.linalg.norm(half))
        # Support of -z at rotation phi is 0.5 |sin phi| + 0.1 |cos phi|; it peaks near phi = pi/2.
        limit = 0.3
        angle = math.pi  # end support 0.1 < limit, peak 0.5 > limit
        out = wp.zeros(1, dtype=float, device="cpu")
        wp.launch(
            _rotating_box_fraction_kernel,
            dim=1,
            inputs=[angle, limit, wp.vec3(*half), lipschitz],
            outputs=[out],
            device="cpu",
        )
        fraction = float(out.numpy()[0])
        self.assertLess(fraction, 1.0)
        phi = np.linspace(0.0, fraction * angle, 20001)
        support = 0.5 * np.abs(np.sin(phi)) + 0.1 * np.abs(np.cos(phi))
        self.assertLessEqual(float(support.max()), limit + 1.0e-6)
        self.assertGreater(fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
