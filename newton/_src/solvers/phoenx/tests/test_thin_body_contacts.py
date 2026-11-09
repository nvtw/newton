# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Thin plates (playing cards) against a ground plane.

A 63.5 x 88.9 mm card is only 0.3-0.7 mm thick, so its corner effective
mass is dominated by rotation and its whole thickness is smaller than one
substep of falling motion. These tests pin three properties that thin
bodies need: the contact manifold spans the face that will land, rows
reused across substeps track the closing gap, and the velocity relaxation
removes the depenetration velocity instead of turning it into spin.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx.tests._test_helpers import require_cuda_graph_capture

CARD_WIDTH, CARD_HEIGHT, CARD_MASS = 0.0635, 0.0889, 0.0018
FRAME_DT = 1.0 / 60.0


def _card_scene(thickness: float, poses, velocities, prepare_refresh_stride: int):
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    cfg = newton.ModelBuilder.ShapeConfig(
        density=CARD_MASS / (CARD_WIDTH * CARD_HEIGHT * thickness), mu=0.8, restitution=0.0
    )
    for pose, velocity in zip(poses, velocities, strict=True):
        body = builder.add_body(xform=pose)
        builder.add_shape_box(body, hx=0.5 * CARD_HEIGHT, hy=0.5 * CARD_WIDTH, hz=0.5 * thickness, cfg=cfg)
        builder.body_qd[-1] = velocity
    model = builder.finalize()
    solver = newton.solvers.SolverPhoenX(
        model,
        step_layout="single_world",
        substeps=8,
        solver_iterations=8,
        prepare_refresh_stride=prepare_refresh_stride,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    pipeline = model._collision_pipeline
    contacts = pipeline.contacts()

    def simulate():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, FRAME_DT)
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)

    with wp.ScopedCapture() as capture:
        simulate()
    return state_0, capture.graph


def _lowest_corner(body_q: np.ndarray, thickness: float) -> float:
    """Lowest card corner height [m] over all cards."""
    lowest = math.inf
    for q in body_q:
        transform = wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:]))
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner = wp.vec3(0.5 * sx * CARD_HEIGHT, 0.5 * sy * CARD_WIDTH, 0.5 * sz * thickness)
                    lowest = min(lowest, float(wp.transform_point(transform, corner)[2]))
    return lowest


class TestThinBodyContacts(unittest.TestCase):
    def setUp(self):
        require_cuda_graph_capture("PhoenX thin-body contact tests")

    def test_penetrating_card_recovers_without_overshoot(self):
        """A flat card started more than its thickness inside the ground rests on it.

        The soft contact pushes the card out over several substeps that reuse
        the prepared rows. Each reused row must see the gap the previous
        substeps already closed; a stale gap keeps pushing at the initial depth
        and launches the card.
        """
        thickness = 0.7e-3
        pose = wp.transform(wp.vec3(0.0, 0.0, -0.5e-3), wp.quat_identity())
        state, graph = _card_scene(thickness, [pose], [(0.0,) * 6], prepare_refresh_stride=3)
        highest = -math.inf
        for _ in range(30):
            wp.capture_launch(graph)
            highest = max(highest, float(state.body_q.numpy()[0, 2]))
        q, qd = state.body_q.numpy()[0], state.body_qd.numpy()[0]
        self.assertLess(highest, 0.5 * thickness + 0.2e-3, f"card overshot to {1e3 * highest:.2f} mm")
        self.assertAlmostEqual(float(q[2]), 0.5 * thickness, delta=0.05e-3)
        self.assertLess(float(np.linalg.norm(qd[:3])), 1.0e-3)
        self.assertLess(float(np.linalg.norm(qd[3:])), 1.0e-2)

    def test_tumbling_cards_land_on_the_plane(self):
        """Tilted, spinning cards dropped onto the ground never sink into it.

        Covers the plane manifold of a tilted plate (the opposite edge must stay
        in the manifold) and contact rows reused across substeps (their gaps must
        close with the motion instead of admitting the same approach again).
        """
        for thickness in (0.3e-3, 0.7e-3):
            with self.subTest(thickness=thickness):
                poses, velocities = [], []
                for k, (height, speed) in enumerate(((0.1, 0.0), (0.1, -1.5), (0.05, -3.0))):
                    axis = wp.normalize(wp.vec3(1.0, 0.3, 0.0))
                    poses.append(
                        wp.transform(
                            wp.vec3(0.15 * k, 0.0, height), wp.quat_from_axis_angle(axis, math.radians(30.0) + 0.4 * k)
                        )
                    )
                    velocities.append((0.0, 0.0, speed, 2.0 * k, 3.0, 0.0))
                state, graph = _card_scene(thickness, poses, velocities, prepare_refresh_stride=3)
                lowest = math.inf
                for _ in range(120):
                    wp.capture_launch(graph)
                    lowest = min(lowest, _lowest_corner(state.body_q.numpy(), thickness))
                self.assertGreater(lowest, -0.4e-3, f"a card corner sank {1e3 * -lowest:.2f} mm into the ground")
                centers = state.body_q.numpy()[:, 2]
                np.testing.assert_allclose(centers, 0.5 * thickness, atol=0.05e-3)


if __name__ == "__main__":
    unittest.main()
