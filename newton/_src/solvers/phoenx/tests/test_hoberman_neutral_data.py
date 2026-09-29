# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Hoberman example is built from embedded body and joint data."""

from __future__ import annotations

import unittest
from collections import Counter
from unittest.mock import patch

import newton
from newton._src.solvers.phoenx.examples.example_hoberman_sphere import make_hoberman_builder


class TestHobermanNeutralData(unittest.TestCase):
    def test_builder_does_not_load_usd(self) -> None:
        with patch.object(newton.ModelBuilder, "add_usd", side_effect=AssertionError("runtime USD import")):
            builder = make_hoberman_builder()

        self.assertEqual(builder.body_count, 240)
        self.assertEqual(builder.joint_count, 421)
        self.assertEqual(Counter(joint.name for joint in builder.joint_type), {"FREE": 1, "REVOLUTE": 360, "BALL": 60})
        self.assertEqual(len(builder.shape_type), 480)
        self.assertEqual(sum(a >= 0 for a in builder.joint_articulation), 240)


if __name__ == "__main__":
    unittest.main()
