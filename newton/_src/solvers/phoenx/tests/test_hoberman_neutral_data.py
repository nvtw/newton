# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Hoberman example is built from embedded body and joint data."""

from __future__ import annotations

import unittest
from collections import Counter
from itertools import product
from math import pi, sin
from unittest.mock import patch

import warp as wp

import newton
from newton._src.solvers.phoenx.examples import hoberman_sphere_data
from newton._src.solvers.phoenx.examples.example_hoberman_sphere import make_hoberman_builder
from newton._src.solvers.phoenx.examples.example_kapla_hoberman import HOBERMAN_COUNT, SPHERE_RING_RADIUS


class TestHobermanNeutralData(unittest.TestCase):
    def test_builder_does_not_load_usd(self) -> None:
        """Build the mechanism entirely from neutral Python data."""
        with patch.object(newton.ModelBuilder, "add_usd", side_effect=AssertionError("runtime USD import")):
            builder = make_hoberman_builder()

        self.assertEqual(builder.body_count, 240)
        self.assertEqual(builder.joint_count, 421)
        self.assertEqual(Counter(joint.name for joint in builder.joint_type), {"FREE": 1, "REVOLUTE": 360, "BALL": 60})
        self.assertEqual(len(builder.shape_type), 480)
        self.assertEqual(sum(a >= 0 for a in builder.joint_articulation), 240)

    def test_mixed_scene_spheres_do_not_overlap(self) -> None:
        """Enclose every shape vertex and require disjoint sphere bounds."""
        radius = 0.0
        for body, pose, half_extents, _color in hoberman_sphere_data.TILES:
            body_pose = hoberman_sphere_data.BODIES[body][1]
            body_xform = wp.transform(wp.vec3(*body_pose[:3]), wp.quat(*body_pose[3:]))
            shape_xform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            for signs in product((-1.0, 1.0), repeat=3):
                corner = wp.vec3(*(sign * extent for sign, extent in zip(signs, half_extents, strict=True)))
                point = wp.transform_point(body_xform * shape_xform, corner)
                radius = max(radius, float(wp.length(point)))
        spacing = 2.0 * SPHERE_RING_RADIUS * sin(pi / HOBERMAN_COUNT)
        # Include both shapes' 1 cm speculative-contact gaps.
        self.assertGreater(spacing, 2.0 * radius + 0.02)


if __name__ == "__main__":
    unittest.main()
