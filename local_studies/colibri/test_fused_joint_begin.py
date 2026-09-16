# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check local fused preparation across live joint property transitions."""

import unittest

import numpy as np

from local_studies.colibri import fused_joint_begin as fused
from local_studies.colibri.test_inactive_joint_prepare import run


class TestFusedBegin(unittest.TestCase):
    def test_live_transitions(self):
        """Preserve state and momentum through finite limits, friction and drive changes."""
        cls = fused.BlockJointSystem
        try:
            for prismatic in (False, True):
                for split in (False, True):
                    cls.__init__ = fused._original_init
                    cls.begin_substep = fused._original_begin
                    expected, masses_expected = run(True, prismatic, split)
                    cls.__init__ = fused._init
                    cls.begin_substep = fused._begin
                    actual, masses_actual = run(True, prismatic, split)
                    np.testing.assert_array_equal(masses_actual, masses_expected)
                    for expected_frame, actual_frame in zip(expected, actual, strict=True):
                        for expected_array, actual_array in zip(expected_frame, actual_frame, strict=True):
                            np.testing.assert_array_equal(actual_array.view(np.uint32), expected_array.view(np.uint32))
        finally:
            cls.__init__ = fused._original_init
            cls.begin_substep = fused._original_begin


if __name__ == "__main__":
    unittest.main()
