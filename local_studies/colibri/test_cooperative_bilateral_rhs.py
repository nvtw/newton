# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Cache lifecycle equivalence through real finite joint-property transitions."""

import unittest

import numpy as np

from local_studies.colibri.cooperative_bilateral_rhs import install
from newton._src.solvers.phoenx.tests import test_inactive_joint_prepare as transitions


class TestCooperativeBilateralRHS(unittest.TestCase):
    def test_property_and_geometry_refresh(self):
        """Keep exact states through motion, finite bounds, friction and row rebinds."""
        original = transitions.make_solver

        def grouped(model, **kwargs):
            return original(model, mass_splitting_color_group_size=2, **kwargs)

        transitions.make_solver = grouped
        try:
            for prismatic in (False, True):
                expected, _ = transitions.run(True, prismatic, True)
                restore = install()
                try:
                    actual, _ = transitions.run(True, prismatic, True)
                finally:
                    restore()
                for before, after in zip(expected, actual, strict=True):
                    for a, b in zip(before, after, strict=True):
                        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        finally:
            transitions.make_solver = original


if __name__ == "__main__":
    unittest.main()
