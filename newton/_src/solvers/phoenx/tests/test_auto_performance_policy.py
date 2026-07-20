# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.solvers.phoenx.solver import _resolve_auto_step_layout


class TestPhoenXAutoPerformancePolicy(unittest.TestCase):
    def _resolve(self, **overrides):
        options = {
            "step_layout": "auto",
            "num_worlds": 1,
            "body_count": 512,
            "has_joints": False,
            "has_deformables": False,
            "has_shapes": True,
            "solver_flavor": "standard",
            "contact_friction_model": "point",
            "articulation_mode": "maximal",
        }
        options.update(overrides)
        return _resolve_auto_step_layout(**options)

    def test_large_rigid_contact_world_selects_single_world(self):
        self.assertEqual(self._resolve(), "single_world")

    def test_noneligible_topologies_keep_multi_world(self):
        for overrides in (
            {"body_count": 511},
            {"num_worlds": 2},
            {"has_joints": True},
            {"has_deformables": True},
        ):
            with self.subTest(**overrides):
                self.assertEqual(self._resolve(**overrides), "multi_world")

    def test_explicit_overrides_are_preserved(self):
        self.assertEqual(self._resolve(step_layout="multi_world"), "multi_world")
        self.assertEqual(self._resolve(step_layout="single_world"), "single_world")


if __name__ == "__main__":
    unittest.main()
